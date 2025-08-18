import argparse
import copy
from copy import deepcopy
from html import parser
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from samplers import euler_maruyama_sampler
from loss import SILoss

from dataset import CustomDataset, ShardWindowSampler
# import wandb_utils
import wandb
from diffusers.models import AutoencoderKL
from PIL import Image
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass

def _worker_init_fn(_):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass

def _warm_shards(ds, shard_ids):
    """Open + touch first row of each selected shard to warm page cache."""
    try:
        for sid in shard_ids:
            ds.vae_arr._ensure_open(sid)[0]             # VAE
            if getattr(ds, "dino_cls", None) is not None:
                ds.dino_cls._ensure_open(sid)[0]        # DINO cls
                ds.dino_pat._ensure_open(sid)[0]        # DINO patches
                if getattr(ds, "dino_regs", None) is not None:
                    ds.dino_regs._ensure_open(sid)[0]   # DINO registers (optional)
    except Exception:
        # Be fail-soft: if internals ever change, training should still run.
        pass

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)



def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs()]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # ------------------------------------------------------------------
    # Dataset setup: either use precomputed DINO features, or compute on-the-fly
    # ------------------------------------------------------------------
    if args.dino_otf:
        # Need raw images & VAE latents only; we'll compute DINO per step.
        train_dataset = CustomDataset(
            args.data_dir,
            load_dinov3=False,
            dinov3_subdir=args.dinov3_subdir,
            load_registers=False,
            load_image=True,
        )
    else:
        train_dataset = CustomDataset(
            args.data_dir,
            load_dinov3=True,
            dinov3_subdir=args.dinov3_subdir,
            load_registers=False,
            load_image=False,
        )

    if args.sampler == "shardwindow":
        sampler = ShardWindowSampler(
            train_dataset,
            window_shards=accelerator.num_processes * args.window_shards,
            strategy='cover',
            seed=args.seed,
            ddp_rank=accelerator.process_index,
            ddp_world=accelerator.num_processes,
        )
    else:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            drop_last=True,
        )

    # Enlarge per-array mmap LRU to match the shard window (reduce reopen thrash)
    try:
        target_cache = sampler.window_shards if hasattr(sampler, "window_shards") else 8
        for arr in [train_dataset.vae_arr,
                    getattr(train_dataset, "dino_cls", None),
                    getattr(train_dataset, "dino_pat", None),
                    getattr(train_dataset, "dino_regs", None)]:
            if arr is not None:
                arr.cache = max(arr.cache, target_cache)
    except Exception:
        pass

    # Warm the first epoch’s shard window (prime OS page cache)
    if args.sampler == "shardwindow":
        w = sampler.window_shards
        cover = sampler._cover
        start = (0 * w) % max(w, sampler.num_shards)
        shard_ids = list(cover[start:start + w])
        if len(shard_ids) < w:
            shard_ids += list(cover[:(w - len(shard_ids))])
        _warm_shards(train_dataset, shard_ids)


    # DINO model / processor if on-the-fly
    dino_model = None
    dino_processor = None
    if args.dino_otf:
        if accelerator.is_main_process:
            logger.info(f"Loading DINOv3 model on rank0: {args.dinov3_model} (dtype={args.dino_dtype})")
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.dino_dtype]
        dino_processor = AutoImageProcessor.from_pretrained(args.dinov3_model)
        dino_model = AutoModel.from_pretrained(args.dinov3_model, torch_dtype=torch_dtype).to(device)
        dino_model.eval()
        if accelerator.is_main_process:
            logger.info("DINOv3 model loaded for on-the-fly encoding.")
        # infer token dim by one forward
        tmp = train_dataset[0]
        if len(tmp) != 3:
            raise RuntimeError("Unexpected dataset return for OTF mode; expected 3 elements (img, vae, label).")
        sample_raw, sample_vae, sample_lab = tmp
        # make a tiny batch
        sample_raw_t = torch.from_numpy(sample_raw) if isinstance(sample_raw, np.ndarray) else sample_raw
        if sample_raw_t.ndim == 3 and sample_raw_t.shape[0] in (1,3):
            # channels-first -> convert to HWC for processor (expects PIL or numpy HWC)
            sample_raw_np = sample_raw_t.permute(1,2,0).cpu().numpy()
        elif sample_raw_t.ndim == 3 and sample_raw_t.shape[-1] in (1,3):
            sample_raw_np = sample_raw_t.cpu().numpy()
        else:
            raise RuntimeError("Cannot parse raw image shape for DINO OTF inference.")
        inputs = dino_processor(images=[sample_raw_np], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
        lhs = outputs.last_hidden_state  # [1, 1+R+N, D]
        token_dim = lhs.shape[-1]
    else:
        # Infer token embedding dim from a single sample (precomputed mode)
        tmp = train_dataset[0]
        if len(tmp) == 5:
            _raw_img, _vae_latents, _lbl, tmp_tokens, _tmp_cls = tmp
            token_dim = tmp_tokens.shape[-1]
        else:
            raise RuntimeError(
                f"Dataset must be initialized with load_dinov3=True returning 5 elements. Got {len(tmp)} elements.")
    z_dims = [token_dim]
    encoders = []  # kept for SILoss interface (unused)
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    # train_dataset already created above
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=(_worker_init_fn if args.num_workers > 0 else None),
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    # In your training script, after creating the sampler, add:
    if accelerator.is_main_process:
        # Test the sampler length
        sampler_len = len(train_dataloader)
        print(f"DataLoader length: {sampler_len}")
        if args.window_shards:
            print(f"Window shards: {sampler.window_shards}")
            print(f"Total shards: {sampler.num_shards}")
            print(f"Sampler counts (first 10): {sampler.counts[:10]}")
            print(f"DDP rank/world: {accelerator.process_index}/{accelerator.num_processes}")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu', weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    # model, optimizer, train_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader
    # )
    model, optimizer = accelerator.prepare(model, optimizer)

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REG",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    sample_batch = next(iter(train_dataloader))
    if args.dino_otf:
        if len(sample_batch) != 3:
            raise RuntimeError("OTF DINO mode expects dataset to return 3 elements.")
        gt_raw_images, gt_xs, _labels = sample_batch
    else:
        if len(sample_batch) == 5:
            gt_raw_images, gt_xs, _labels, _tokens, _cls = sample_batch
        else:
            raise RuntimeError("Precomputed DINO mode expects 5-element batch.")
    # assert gt_raw_images.shape[-1] == args.resolution, "Resolution mismatch with dataset images."
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias)
    ys = torch.randint(0, args.num_classes, size=(sample_batch_size,), device=device)
    # Create sampling noise for image latents:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    # cls latents (dimension inferred from token_dim / cls token size)
    cls_latent_dim = token_dim
    cls_z = torch.randn(n, 5, cls_latent_dim, device=device)

    # sampling directory & VAE placeholder
    if accelerator.is_main_process:
        sample_dir = os.path.join(args.output_dir, args.exp_name, "samples")
        os.makedirs(sample_dir, exist_ok=True)
    vae = None  # lazily loaded
        
    for epoch in range(args.epochs):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        for batch in train_dataloader:
            if args.dino_otf:
                if len(batch) != 3:
                    raise RuntimeError("OTF DINO mode expects 3-element batch (img, vae, label).")
                raw_image, x, y = batch
                # raw_image: tensor (B, H, W, C) or (B, C, H, W)
                if raw_image.ndim != 4:
                    raise RuntimeError("Raw image batch has unexpected ndim.")
                if raw_image.shape[1] in (1,3) and raw_image.shape[-1] not in (1,3):
                    # channels-first -> convert to list of HWC numpy arrays
                    imgs_np = raw_image.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
                elif raw_image.shape[-1] in (1,3):
                    imgs_np = raw_image.cpu().numpy().astype(np.uint8)
                else:
                    raise RuntimeError("Cannot determine channel placement for raw images.")
                inputs = dino_processor(images=list(imgs_np), return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                lhs = outputs.last_hidden_state  # [B, 1+R+N, D]
                R = dino_model.config.num_register_tokens
                dinov3_cls = lhs[:,0:R,:]
                # skip register tokens for now (consistent with precomputed path w/o registers)
                dinov3_tokens = lhs[:,1+R:,:]
            else:
                if len(batch) != 5:
                    raise RuntimeError("Precomputed DINO mode expects 5-element batch.")
                raw_image, x, y, dinov3_tokens, dinov3_cls = batch
            x = x.squeeze(1).to(device, non_blocking=True)  # VAE latents (mean+std packed)
            y = y.to(device, non_blocking=True)
            dinov3_tokens = dinov3_tokens.to(device, non_blocking=True)  # [B, 1+N, D]
            dinov3_cls = dinov3_cls.to(device, non_blocking=True)        # [B, D]

            if args.legacy:
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            cls_token = dinov3_cls
            if cls_token.ndim == 2:
                cls_token = cls_token.unsqueeze(1)  # (B, 1, D)

            zs = [torch.cat([cls_token, dinov3_tokens], dim=1)]

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss1, proj_loss1, time_input, noises, loss2 = loss_fn(
                    model, x, model_kwargs,
                    zs=zs,
                    cls_token=cls_token,
                    time_input=None, noises=None
                )
                loss_mean = loss1.mean()
                loss_mean_cls = loss2.mean() * args.cls
                proj_loss_mean = proj_loss1.mean() * args.proj_coeff
                loss = loss_mean + proj_loss_mean + loss_mean_cls
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, model)
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # ------------------------------------------------------------
            # In-training sampling (EMA model) every args.sampling_steps
            # Log image grid to wandb every 20000 steps
            # ------------------------------------------------------------
            if (global_step % args.sampling_steps == 0) and global_step > 0:
                with torch.no_grad():
                    ema.eval()
                    # load VAE once
                    if vae is None:
                        try:
                            vae_local = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
                        except Exception as e:
                            logging.warning(f"Failed to load VAE: {e}; skipping sampling this step.")
                            vae_local = None
                        vae = vae_local
                    if vae is not None:
                        sampling_kwargs = dict(
                            model=ema,
                            latents=xT.clone(),
                            y=ys.clone(),
                            num_steps=args.num_sample_steps,
                            heun=False,
                            cfg_scale=args.cfg_scale,
                            guidance_low=args.guidance_low,
                            guidance_high=args.guidance_high,
                            path_type=args.path_type,
                            cls_latents=cls_z.clone(),
                            args=args,
                        )

                        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                        # decode to pixels
                        samples = vae.decode((samples - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        samples = samples.clamp(0, 1)
                        accelerator.wait_for_everyone()
                        gathered = accelerator.gather(samples)
                        if accelerator.is_main_process:
                            grid = array2grid(gathered)
                            Image.fromarray(grid).save(f"{sample_dir}/samples_step_{global_step}.png")
                            logger.info(f"Saved samples at step {global_step}")
                            if global_step % 20000 == 0:
                                accelerator.log({"samples": wandb.Image(grid)}, step=global_step)


            if accelerator.sync_gradients:
                logs = {
                    "loss_final": accelerator.gather(loss).mean().detach().item(),
                    "loss_mean": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "loss_mean_cls": accelerator.gather(loss_mean_cls).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                }
            else:
                logs = {}

            # log_message = ", ".join(f"{key}: {value:.6f}" for key, value in logs.items())
            # logging.info(f"Step: {global_step}, Training Logs: {log_message}")

            if logs:
                progress_bar.set_postfix(**{k: v for k, v in logs.items() if k.startswith('loss')})
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=1000, help="Frequency (in steps) to run EMA sampling during training.")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ops-head", type=int, default=16)

    # dataset
    # add to argparse (near dataset args)
    parser.add_argument("--window-shards", type=int, default=8)
    parser.add_argument("--sampler", type=str, choices=["distributed","shardwindow"], default="shardwindow")
    parser.add_argument("--data-dir", type=str, default="../dataset")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=8)#256

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1000000)
    parser.add_argument("--max-train-steps", type=int, default=2400001)
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cls", type=float, default=0.03)
    # sampling specific
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale for in-training sampling.")
    parser.add_argument("--cls-cfg-scale", type=float, default=1.0, help="CLS guidance scale (used inside sampler).")
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--num-sample-steps", type=int, default=50, help="Diffusion sampling steps for in-training sampling.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse", help="Which Stable Diffusion VAE variant to use for decoding samples.")
    # DINO on-the-fly options
    parser.add_argument("--dino-otf", action=argparse.BooleanOptionalAction, default=False, help="Compute DINOv3 embeddings on-the-fly instead of using precomputed shards.")
    parser.add_argument("--dinov3-model", type=str, default="facebook/dinov3-vit7b16-pretrain-lvd1689m", help="DINOv3 HF model id for OTF mode.")
    parser.add_argument("--dinov3-subdir", type=str, default="dinov3-vit7b16", help="Subdirectory name for precomputed DINO features (when not OTF).")
    parser.add_argument("--dino-dtype", type=str, choices=["float32","float16","bfloat16"], default="float16")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
