# dataset_tools.py
# Robust, aligned, memory-mapped shards for images, VAE moments (8ch), and DINOv3 tokens.
# Out-of-order decode/encode; single-writer main process writes directly to shard offsets (via ShardMap).
#
# Key features vs. previous version:
# - Index→shard mapping via offsets (ShardMap). No rps assumptions anywhere.
# - Periodic verification of recently written VAE/DINO rows + final strict FSCK with hard failure if missing.
# - DINO inputs normalized using AutoImageProcessor config (rescale + mean/std).
# - Explicit memmap flushes during and after.
#
# Typical usage:
#   python dataset_tools.py images \
#       --source /path/imagenet --dest DATA --resolution 256x256 --transform center-crop-dhariwal
#
#   python dataset_tools.py encode-vae  --dataset-root DATA --gpus 8 --batch-size 128
#   python dataset_tools.py encode-dino --dataset-root DATA --gpus 8 --batch-size 64 \
#       --model-name facebook/dinov3-vit7b16-pretrain-lvd1689m --dtype float16 --dino-dim 4096 \
#       --patch-size 16 --num-register 4
#
#   # End-to-end in one command (recommended):
#   python dataset_tools.py all \
#       --source /path/imagenet --dest DATA --resolution 256x256 --transform center-crop-dhariwal \
#       --gpus 8 --batch-size 64 --vae-url stabilityai/sd-vae-ft-mse \
#       --dino-model facebook/dinov3-vit7b16-pretrain-lvd1689m --dino-dtype float16 \
#       --dino-dim 4096 --patch-size 16 --num-register 4
#
# Inspection:
#   python dataset_tools.py fsck  --dataset-root DATA [--group images|vae|dino|all] [--strict] [--jobs N]
#   python dataset_tools.py stats --dataset-root DATA
#
# Layout:
#   DATA/
#     images/packed/{meta.json, labels.npy, order_keys.txt, images_00000.npy, ...}
#     vae-sd/packed/{meta.json, labels.npy, order_keys.txt, vae_latents_00000.npy, ...}
#     dinov3-vit7b16/packed/{meta.json, labels.npy, order_keys.txt,
#                            dinov3_cls_00000.npy, dinov3_registers_00000.npy, dinov3_patches_00000.npy, ...}

import os, re, io, json, math, zipfile, multiprocessing as mp, time, queue, threading, glob, random
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator, List, Dict, Any
from pathlib import Path
from bisect import bisect_right

import click
import numpy as np
from tqdm import tqdm
import PIL.Image
import PIL.ImageFile

import torch
from transformers import AutoModel, AutoImageProcessor

# your local encoder wrapper
from encoders import StabilityVAEEncoder

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid PIL hangs on truncated files


# --------------------------
# Helpers / common utilities
# --------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _open_memmap(path: str, shape, dtype):
    _ensure_dir(os.path.dirname(path))
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)

def _save_json(path: str, obj: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f: json.dump(obj, f)

def _load_json(path: str) -> dict:
    with open(path, 'r') as f: return json.load(f)

def _write_lines(path: str, lines: List[str]):
    _ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        for s in lines: f.write(s + '\n')

def _parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if not m:
        raise click.ClickException(f'cannot parse WxH tuple: {s}')
    return int(m.group(1)), int(m.group(2))

def _is_image_ext(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in PIL.Image.EXTENSION

def _byte_size(shape, dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize

def _records_per_shard(target_bytes: int, sample_shape, dtype, floor_min: int = 64) -> int:
    sz = _byte_size(sample_shape, dtype)
    return max(floor_min, target_bytes // max(1, sz))

def _labels_from_dirs(source_root: str, relpaths: List[str]) -> np.ndarray:
    # Honor dataset.json if present
    dj = os.path.join(source_root, 'dataset.json')
    if os.path.isfile(dj):
        with open(dj, 'r') as f:
            data = json.load(f).get('labels')
        if data is None:
            return np.zeros(len(relpaths), dtype=np.int64)
        mapping = {p: lab for (p, lab) in data}
        labs = [mapping.get(p.replace('\\','/'), 0) for p in relpaths]
        return np.asarray(labs, dtype=np.int64)

    tops = [rp.split('/')[0] if '/' in rp else '' for rp in relpaths]
    uniq = sorted(set(tops))
    if len(uniq) <= 1:
        return np.zeros(len(relpaths), dtype=np.int64)
    lut = {name: i for i, name in enumerate(uniq)}
    return np.asarray([lut[t] for t in tops], dtype=np.int64)


# --------------------------
# Index/Shard mapping (no rps assumptions)
# --------------------------

class ShardMap:
    def __init__(self, paths: List[str], counts: List[int]):
        self.paths = paths
        self.counts = [int(c) for c in counts]
        self.offsets = [0]
        acc = 0
        for c in self.counts:
            acc += int(c)
            self.offsets.append(acc)  # len = shards + 1
        self.total = self.offsets[-1]
        # sanity: strictly increasing
        for i in range(1, len(self.offsets)):
            if not (self.offsets[i] > self.offsets[i-1]):
                raise ValueError(f"non-increasing offsets at i={i}: {self.offsets[i-1]} -> {self.offsets[i]}")

    def locate(self, gidx: int) -> Tuple[int, int]:
        gi = int(gidx)
        if gi < 0:
            raise IndexError(f"global index {gi} out of range (<0)")
        if gi >= self.total:
            raise IndexError(f"global index {gi} out of range (>= total {self.total})")

        # Find sid such that offsets[sid] <= gi < offsets[sid+1]
        lo, hi = 0, len(self.counts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if gi < self.offsets[mid]:
                hi = mid - 1
            elif gi >= self.offsets[mid + 1]:
                lo = mid + 1
            else:
                sid = mid
                break
        else:
            raise IndexError(f"could not locate global index {gi} in offsets={self.offsets}")

        off = gi - self.offsets[sid]
        if not (0 <= off < self.counts[sid]):
            raise IndexError(
                f"computed offset {off} out of range in shard {sid} "
                f"(count={self.counts[sid]}, gi={gi}, offsets[{sid}]={self.offsets[sid]}, offsets[{sid+1}]={self.offsets[sid+1]})"
            )
        return sid, off



@dataclass
class ShardedWriter:
    map: ShardMap
    writers: List[np.memmap]

    def __post_init__(self):
        # Ensure map counts match the real memmap first-dimension sizes.
        counts_from_writers = [int(w.shape[0]) for w in self.writers]
        if (len(counts_from_writers) != len(self.map.counts)) or (counts_from_writers != list(self.map.counts)):
            # Rebuild the map using actual writer sizes; preserve path order.
            print("[ShardMap] counts mismatch with writers; rebuilding map from writer shapes.")
            self.map = ShardMap(self.map.paths, counts_from_writers)

    def write_batch(self, gidxs: np.ndarray, batch: np.ndarray):
        if batch.shape[0] != gidxs.shape[0]:
            raise ValueError(f"[ShardWriter] batch/gidx length mismatch: {batch.shape[0]} vs {gidxs.shape[0]}")

        for i, g in enumerate(gidxs.tolist()):
            sid, off = self.map.locate(int(g))
            w = self.writers[sid]
            if not (0 <= off < w.shape[0]):
                _dump = {
                    "gidx": g, "sid": sid, "off": off, "w_shape": tuple(w.shape),
                    "map_counts_window": self.map.counts[max(0, sid-3):sid+4],
                    "map_offsets_window": self.map.offsets[max(0, sid-3):sid+5],
                    "first_g": int(gidxs[0]), "last_g": int(gidxs[-1]),
                }
                print(f"[ShardWriter][OOB] {json.dumps(_dump)}", flush=True)
                if off == w.shape[0] and sid + 1 < len(self.writers):
                    sid += 1
                    w = self.writers[sid]
                    off = 0
                else:
                    raise IndexError(f"[ShardWriter] OOB write: sid={sid} off={off} shard0={w.shape[0]} g={g}")
            w[off] = batch[i]

    def flush(self):
        for mm in self.writers:
            try: mm.flush()
            except Exception: pass

    def close(self):
        for mm in self.writers:
            del mm


# --------------------------
# Periodic reporter & verifier
# --------------------------

class PeriodicReporter(threading.Thread):
    def __init__(self, snapshot_fn, interval=10.0, name="monitor"):
        super().__init__(daemon=True)
        self.snapshot_fn = snapshot_fn
        self.interval = interval
        self._stop = threading.Event()
        self.name = name

    def run(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            try:
                snap = self.snapshot_fn()
                print(f"[{self.name}] " + " ".join(f"{k}={v}" for k,v in snap.items()))
            except Exception as e:
                print(f"[{self.name}] snapshot error: {e}")

    def stop(self):
        self._stop.set()


class PeriodicVerifier(threading.Thread):
    def __init__(self, sample_fn, check_fn, interval=20.0, sample_k=256, name="verifier"):
        super().__init__(daemon=True)
        self.sample_fn = sample_fn
        self.check_fn  = check_fn
        self.interval  = interval
        self.sample_k  = sample_k
        self._stop     = threading.Event()
        self.name      = name
        self.error     = None

    def run(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            try:
                sample = self.sample_fn(self.sample_k)
                bad = []
                for stream, items in sample.items():
                    for path, row_off, shape in items:
                        ok = self.check_fn(path, row_off, shape)
                        if not ok:
                            bad.append((stream, path, row_off))
                if bad:
                    msgs = "\n".join([f"  {s}: {p} [{r}]" for (s,p,r) in bad])
                    raise RuntimeError(f"[{self.name}] detected zero rows during run:\n{msgs}")
                else:
                    print(f"[{self.name}] spot-check OK on {sum(len(v) for v in sample.values())} rows.")
            except Exception as e:
                # Don’t just crash the thread; record error and stop so main can raise.
                self.error = e
                print(f"[{self.name}] ERROR: {e}")
                self._stop.set()

    def stop(self):
        self._stop.set()

# --------------------------
# Enumerate image sources
# --------------------------

@dataclass
class ImgRef:
    kind: str                # 'dir' or 'zip'
    abs_path: str            # absolute path (dir root or zip file path)
    inner_rel: Optional[str] # file path for 'dir' mode; inner path for 'zip' mode

def _walk_dir(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if _is_image_ext(fn):
                rel = os.path.relpath(os.path.join(r, fn), root).replace('\\','/')
                out.append(rel)
    out.sort()
    return out

def _walk_zip(zpath: str) -> List[str]:
    out = []
    with zipfile.ZipFile(zpath, 'r') as z:
        for name in sorted(z.namelist()):
            if _is_image_ext(name):
                out.append(name)
    return out

def _enumerate_images(source: str) -> Tuple[List[ImgRef], List[str], str]:
    if os.path.isdir(source):
        rels = _walk_dir(source)
        refs = [ImgRef('dir', os.path.abspath(source), r) for r in rels]
        return refs, rels, os.path.abspath(source)
    if os.path.isfile(source) and source.lower().endswith('.zip'):
        zabs = os.path.abspath(source)
        rels = _walk_zip(zabs)
        refs = [ImgRef('zip', zabs, r) for r in rels]
        return refs, rels, os.path.dirname(zabs)
    raise click.ClickException(f"Unsupported --source: {source}")

def _open_pil(ref: ImgRef) -> PIL.Image.Image:
    if ref.kind == 'dir':
        return PIL.Image.open(os.path.join(ref.abs_path, ref.inner_rel)).convert('RGB')
    with zipfile.ZipFile(ref.abs_path, 'r') as z:
        with z.open(ref.inner_rel, 'r') as f:
            return PIL.Image.open(f).convert('RGB')


# --------------------------
# Transforms (deterministic)
# --------------------------

def _scale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h: return img
    return np.array(PIL.Image.fromarray(img, 'RGB').resize((w,h), PIL.Image.Resampling.LANCZOS))

def _center_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    crop = min(img.shape[0], img.shape[1])
    img2 = img[(img.shape[0]-crop)//2:(img.shape[0]+crop)//2,
               (img.shape[1]-crop)//2:(img.shape[1]+crop)//2]
    return np.array(PIL.Image.fromarray(img2, 'RGB').resize((w,h), PIL.Image.Resampling.LANCZOS))

def _center_crop_dhariwal(arr: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = PIL.Image.fromarray(arr)
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return arr[cy:cy+image_size, cx:cx+image_size]


# --------------------------
# Shape calculators
# --------------------------

def _vae_moments_shape(H: int, W: int) -> Tuple[int,int,int]:
    # SD VAE moments: [mean, std] packed ⇒ 8 channels; latent is 8x downsampled
    assert H % 8 == 0 and W % 8 == 0, "Resolution must be divisible by 8 for SD VAE."
    return (8, H//8, W//8)

def _dino_shapes(H: int, W: int, dino_dim: int, patch: int, num_register: int):
    assert H % patch == 0 and W % patch == 0, f"Resolution must be divisible by patch size ({patch})."
    N = (H // patch) * (W // patch)      # patch tokens
    D = dino_dim
    cls_shape = (D,)
    regs_shape = (num_register, D)
    patches_shape = (N, D)
    return cls_shape, regs_shape, patches_shape, N


# --------------------------
# Queue message helpers
# --------------------------

def _is_heartbeat(msg) -> bool:
    return isinstance(msg, tuple) and len(msg) >= 2 and isinstance(msg[0], str) and msg[0] in ('ready', 'first-batch')

def _is_error(msg) -> bool:
    return isinstance(msg, tuple) and len(msg) == 3 and isinstance(msg[0], str) and msg[0] == 'err'

def _is_payload_vae(msg) -> bool:
    # (indices[np.int64 B], vae_b[np.float32 Bx8xhxw])
    return isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], np.ndarray)

def _is_payload_dino(msg) -> bool:
    # (indices[np.int64 B], cls[np], regs[np], pats[np])
    return isinstance(msg, tuple) and len(msg) == 4 and isinstance(msg[0], np.ndarray)

def _is_payload_all(msg) -> bool:
    # (indices[np.int64 B], vae[np], cls[np], regs[np], pats[np])
    return isinstance(msg, tuple) and len(msg) == 5 and isinstance(msg[0], np.ndarray)


# --------------------------
# Processing workers
# --------------------------

def _img_worker(args):
    # args: (global_idx, ImgRef, w, h, transform)
    idx, ref, w, h, transform = args
    img = _open_pil(ref)  # let exceptions bubble → fail fast
    arr = np.array(img, dtype=np.uint8)
    if transform is None:
        out = _scale(arr, w, h)
    elif transform == 'center-crop':
        out = _center_crop(arr, w, h)
    elif transform == 'center-crop-dhariwal':
        assert w == h, "Dhariwal crop requires square WxH"
        out = _center_crop_dhariwal(arr, w)
    else:
        raise RuntimeError(f"Unknown transform {transform}")
    return (idx, out)

def _start_image_pool(n_workers: int):
    ctx = mp.get_context('spawn')
    return ctx.Pool(processes=n_workers)

def gpu_worker_vae(gid: int, inq, outq, model_url: str):
    try:
        os.environ.setdefault("OMP_NUM_THREADS","1")
        os.environ.setdefault("MKL_NUM_THREADS","1")
        torch.cuda.set_device(gid)
        dev = torch.device(f'cuda:{gid}')
        enc = StabilityVAEEncoder(vae_name=model_url, batch_size=1)  # internal batching
        outq.put(('ready', gid, 'vae'))
        first = True
        with torch.inference_mode():
            while True:
                item = inq.get()
                if item is None: break
                indices, np_batch = item
                t = torch.from_numpy(np_batch).to(dev, non_blocking=True).permute(0,3,1,2).float()
                moments = enc.encode_pixels(t)[0]  # [B,8,H/8,W/8] float32

                if first:
                    outq.put(('first-batch', gid, int(indices[0]), len(indices)))
                    first = False
                outq.put((indices, moments.detach().cpu().numpy()))
    except Exception as e:
        outq.put(('err', gid, f'{type(e).__name__}: {e}'))

def gpu_worker_dino(gid: int, inq, outq, model_name: str, torch_dtype_name: str,
                    dino_dim: int, patch: int, num_register: int, H: int, W: int):
    try:
        os.environ.setdefault("OMP_NUM_THREADS","1")
        os.environ.setdefault("MKL_NUM_THREADS","1")
        torch.cuda.set_device(gid)
        dev = torch.device(f'cuda:{gid}')
        torch_dtype = getattr(torch, torch_dtype_name)
        mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(dev).eval()
        proc = AutoImageProcessor.from_pretrained(model_name)

        # precompute normalization params
        do_rescale = getattr(proc, "do_rescale", True)
        rescale_factor = getattr(proc, "rescale_factor", 1/255.0 if do_rescale else 1.0)
        do_normalize = getattr(proc, "do_normalize", True)
        mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), device=dev, dtype=torch_dtype).view(1,3,1,1)
        std  = torch.tensor(getattr(proc, "image_std",  [0.229, 0.224, 0.225]), device=dev, dtype=torch_dtype).view(1,3,1,1)

        outq.put(('ready', gid, 'dino'))
        first = True
        with torch.inference_mode():
            while True:
                item = inq.get()
                if item is None: break
                indices, np_batch = item  # np.int64[B], uint8[B,H,W,3]
                t = torch.from_numpy(np_batch).to(dev, non_blocking=True).permute(0,3,1,2).to(torch.float32)
                t = t * rescale_factor
                t = t.to(torch_dtype)
                if do_normalize:
                    t = (t - mean) / std

                h = mdl(pixel_values=t, output_hidden_states=False).last_hidden_state  # [B, 1+R+N, D]
                B, T, D = h.shape
                expected_N = (H // patch) * (W // patch)
                expected_T = 1 + num_register + expected_N
                if T != expected_T:
                    raise RuntimeError(f"DINO token len mismatch (got {T}, expected {expected_T})")
                if D != dino_dim:
                    raise RuntimeError(f"DINO dim mismatch (got {D}, expected {dino_dim})")
                cls  = h[:,0]                           # [B, D]
                regs = h[:,1:1+num_register]           # [B, R, D]
                pats = h[:,1+num_register:]            # [B, N, D]
                out_np_dtype = np.float16 if torch_dtype_name in ('float16','bfloat16') else np.float32
                cls_np  = cls.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                regs_np = regs.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                pats_np = pats.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                if first:
                    outq.put(('first-batch', gid, int(indices[0]), len(indices)))
                    first = False
                outq.put((indices, cls_np, regs_np, pats_np))
    except Exception as e:
        outq.put(('err', gid, f'{type(e).__name__}: {e}'))

def gpu_worker_all(gid: int, inq, outq, vae_url: str,
                   model_name: str, torch_dtype_name: str,
                   dino_dim: int, patch: int, num_register: int, H: int, W: int):
    try:
        os.environ.setdefault("OMP_NUM_THREADS","1")
        os.environ.setdefault("MKL_NUM_THREADS","1")
        torch.cuda.set_device(gid)
        dev = torch.device(f'cuda:{gid}')
        vae = StabilityVAEEncoder(vae_name=vae_url, batch_size=1)
        torch_dtype = getattr(torch, torch_dtype_name)
        mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(dev).eval()
        proc = AutoImageProcessor.from_pretrained(model_name)
        do_rescale = getattr(proc, "do_rescale", True)
        rescale_factor = getattr(proc, "rescale_factor", 1/255.0 if do_rescale else 1.0)
        do_normalize = getattr(proc, "do_normalize", True)
        mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), device=dev, dtype=torch_dtype).view(1,3,1,1)
        std  = torch.tensor(getattr(proc, "image_std",  [0.229, 0.224, 0.225]), device=dev, dtype=torch_dtype).view(1,3,1,1)

        outq.put(('ready', gid, 'all'))
        first = True
        with torch.inference_mode():
            while True:
                item = inq.get()
                if item is None: break
                indices, np_batch = item  # np.int64[B], uint8[B,H,W,3]
                t_f32 = torch.from_numpy(np_batch).to(dev, non_blocking=True).permute(0,3,1,2).float()
                moments = vae.encode_pixels(t_f32)   # [B,8,H/8,W/8] float32

                t = (t_f32 * rescale_factor).to(torch_dtype)
                if do_normalize:
                    t = (t - mean) / std
                h = mdl(pixel_values=t, output_hidden_states=False).last_hidden_state
                B, T, D = h.shape
                expected_N = (H // patch) * (W // patch)
                expected_T = 1 + num_register + expected_N
                if T != expected_T:
                    raise RuntimeError(f"DINO token len mismatch (got {T}, expected {expected_T})")
                if D != dino_dim:
                    raise RuntimeError(f"DINO dim mismatch (got {D}, expected {dino_dim})")
                cls  = h[:,0]
                regs = h[:,1:1+num_register]
                pats = h[:,1+num_register:]
                out_np_dtype = np.float16 if torch_dtype_name in ('float16','bfloat16') else np.float32
                cls_np  = cls.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                regs_np = regs.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                pats_np = pats.detach().to(torch.float16 if out_np_dtype==np.float16 else torch.float32).cpu().numpy()
                if first:
                    outq.put(('first-batch', gid, int(indices[0]), len(indices)))
                    first = False
                outq.put((indices,
                          moments.detach().cpu().numpy(),
                          cls_np, regs_np, pats_np))
    except Exception as e:
        outq.put(('err', gid, f'{type(e).__name__}: {e}'))


# --------------------------
# Click CLI
# --------------------------

@click.group()
def cli():
    PIL.Image.init()


# --------------------------
# Stage 1: IMAGES → shards (single writer with ShardMap)
# --------------------------

@cli.command()
@click.option('--source', required=True, type=str, help='Image folder or .zip')
@click.option('--dest',   required=True, type=str, help='Dataset root (will create subdirs)')
@click.option('--transform', type=click.Choice(['center-crop','center-crop-dhariwal']), default='center-crop-dhariwal', show_default=True)
@click.option('--resolution', required=True, type=str, help='WxH, e.g., 256x256 or 512x512')
@click.option('--shuffle/--no-shuffle', default=True, show_default=True)
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--target-shard-bytes', type=int, default=2_000_000_000, show_default=True)
@click.option('--workers', type=int, default=32, show_default=True)
def images(source, dest, transform, resolution, shuffle, seed, target_shard_bytes, workers):
    """Decode + resize images and write packed uint8 shards with labels & order."""
    w, h = _parse_tuple(resolution)
    refs, rels, label_root = _enumerate_images(source)
    n = len(refs)
    if n == 0: raise click.ClickException("No images found")

    labels = _labels_from_dirs(source if os.path.isdir(source) else os.path.dirname(source), rels)
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(order)

    out_dir = os.path.join(dest, 'images', 'packed')
    _ensure_dir(out_dir)
    np.save(os.path.join(out_dir, 'labels.npy'), labels[order])
    _write_lines(os.path.join(out_dir, 'order_keys.txt'), [f"{i:08d}" for i in order])

    img_shape = (h, w, 3)
    img_dtype = np.uint8
    rps = _records_per_shard(target_shard_bytes, img_shape, img_dtype)
    shard_count = math.ceil(n / rps)

    # open writers
    writers, counts, paths = [], [], []
    for sid in range(shard_count):
        cnt = rps if sid < shard_count-1 else (n - rps*(shard_count-1))
        spath = os.path.join(out_dir, f"images_{sid:05d}.npy")
        writers.append(_open_memmap(spath, shape=(cnt, *img_shape), dtype=img_dtype))
        counts.append(int(cnt))
        paths.append(spath)
    img_map = ShardMap(paths, counts)
    img_writer = ShardedWriter(img_map, writers)

    meta = {"num_samples": int(n), "shape": img_shape, "dtype": str(img_dtype),
            "records_per_shard": int(rps),
            "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(paths, counts)]}

    # launch decode
    decoded_written = 0
    reporter = PeriodicReporter(lambda: dict(decoded=decoded_written, total=n), interval=10.0, name="images-progress")
    reporter.start()
    with _start_image_pool(workers) as pool:
        pbar = tqdm(total=n, desc="Decode/resize + write")
        try:
            CHUNK = 4096
            for i0 in range(0, n, CHUNK):
                batch = [(i, refs[order[i]], w, h, transform) for i in range(i0, min(n, i0+CHUNK))]
                for idx, arr in pool.imap_unordered(_img_worker, batch, chunksize=64):
                    img_writer.write_batch(np.asarray([idx], dtype=np.int64), np.asarray([arr], dtype=np.uint8))
                    decoded_written += 1
                    if (decoded_written % 8192) == 0: img_writer.flush()
                    pbar.update(1)
        finally:
            pbar.close()
            reporter.stop()

    img_writer.flush(); img_writer.close()
    _save_json(os.path.join(out_dir, 'meta.json'), meta)
    print(f"[images] wrote {len(paths)} shards to {out_dir}")


# --------------------------
# Stage 2: VAE from image shards (aligned counts; mapped writes)
# --------------------------

def _images_meta(dataset_root: str) -> dict:
    p = os.path.join(dataset_root, 'images', 'packed', 'meta.json')
    if not os.path.exists(p): raise click.ClickException("images/packed/meta.json not found. Run 'images' or 'all' first.")
    return _load_json(p)

def _iter_image_batches(dataset_root: str, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """yield (indices[int64 B], batch[uint8 BxHxWx3]) across shards, in-order by global index."""
    root = os.path.join(dataset_root, 'images', 'packed')
    meta = _load_json(os.path.join(root, 'meta.json'))
    shards = [os.path.join(root, s["file"]) for s in meta["shards"]]
    counts = [int(s["count"]) for s in meta["shards"]]
    img_map = ShardMap(shards, counts)
    offset = 0
    for sid, spath in enumerate(shards):
        mm = np.load(spath, mmap_mode='r')
        N = mm.shape[0]
        for i in range(0, N, batch_size):
            b = min(batch_size, N - i)
            idxs = np.arange(img_map.offsets[sid] + i, img_map.offsets[sid] + i + b, dtype=np.int64)
            yield idxs, mm[i:i+b]
        offset += N

@cli.command(name='encode-vae')
@click.option('--dataset-root', required=True, type=str)
@click.option('--gpus', type=int, default=8, show_default=True)
@click.option('--batch-size', type=int, default=128, show_default=True)
@click.option('--model-url', type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
def encode_vae(dataset_root, gpus, batch_size, model_url):
    """Read image shards and write aligned VAE moment shards (8ch)."""
    img_meta = _images_meta(dataset_root)
    n = int(img_meta["num_samples"])
    H, W, _ = img_meta["shape"]
    rps = int(img_meta["records_per_shard"])
    vae_shape = _vae_moments_shape(H, W)   # (8, H/8, W/8)
    vae_dtype = np.float32

    out_dir = os.path.join(dataset_root, 'vae-sd', 'packed')
    _ensure_dir(out_dir)
    # share labels/order
    for fname in ('labels.npy','order_keys.txt'):
        src = os.path.join(dataset_root, 'images', 'packed', fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil; shutil.copy(src, dst)

    # open writers (match counts from images)
    paths, writers = [], []
    for sid, s in enumerate(img_meta["shards"]):
        cnt = int(s["count"])
        spath = os.path.join(out_dir, f"vae_latents_{sid:05d}.npy")
        mm = _open_memmap(spath, shape=(cnt, *vae_shape), dtype=vae_dtype)
        # Early sanity: this would catch the "axis 0 is 8" problem immediately
        if mm.shape[0] != cnt or mm.shape[1:] != tuple(vae_shape):
            raise RuntimeError(f"[init-vae] memmap shape mismatch: got {mm.shape}, expected {(cnt, *vae_shape)} at {spath}")
        paths.append(spath); writers.append(mm)

    vae_counts = [int(w.shape[0]) for w in writers]  # source of truth
    vae_map = ShardMap(paths, vae_counts)
    vae_writer = ShardedWriter(vae_map, writers)

    print(f"[init-vae] first shard shape={writers[0].shape} expected=(B, {vae_shape})", flush=True)

    meta = {"num_samples": n, "shape": vae_shape, "dtype": str(vae_dtype),
        "records_per_shard": int(rps),
        "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(paths, vae_counts)]}

    # gpu pool
    ctx = mp.get_context('spawn')
    inqs = [ctx.Queue(maxsize=8) for _ in range(gpus)]
    outq = ctx.Queue(maxsize=64)
    procs = []
    for gid in range(gpus):
        p = ctx.Process(target=gpu_worker_vae, args=(gid, inqs[gid], outq, model_url))
        p.start(); procs.append(p)

    dispatched = 0
    gpu_written = 0
    workers_ready = set()
    first_batch_seen = set()

    # verification bookkeeping
    recent_rows: List[Tuple[str,int,Tuple[int,...]]] = []  # (path, row_off, shape)

    def sample_recent(k):
        random.shuffle(recent_rows)
        out = {"vae": recent_rows[:k]}
        return out

    def check_row(path, row_off, shape):
        mm = np.load(path, mmap_mode='r')
        x = mm[row_off]
        ok = not np.allclose(x, 0.0, atol=0.0)
        del mm
        return ok

    verifier = PeriodicVerifier(sample_fn=sample_recent, check_fn=check_row, interval=20.0, sample_k=128, name="verify-vae")
    verifier.start()

    reporter = PeriodicReporter(
        lambda: dict(dispatched=dispatched, gpu_written=gpu_written, ready=len(workers_ready), first_batches=len(first_batch_seen), total=n),
        interval=10.0, name="vae-progress"
    )
    reporter.start()

    # dispatch + drain
    pbar = tqdm(total=n, desc="VAE encode/write")
    rr = 0
    try:
        for idxs, batch in _iter_image_batches(dataset_root, batch_size):
            inqs[rr % gpus].put((idxs, batch))
            rr += 1
            dispatched += len(idxs)
            # drain opportunistically
            while True:
                try:
                    msg = outq.get_nowait()
                except queue.Empty:
                    break
                if _is_heartbeat(msg):
                    kind, gid = msg[0], msg[1]
                    if kind == 'ready':
                        workers_ready.add(gid); print(f"[vae] gpu{gid} ready")
                    else:
                        first_batch_seen.add(gid); print(f"[vae] gpu{gid} first batch: idx0={msg[2]} size={msg[3]}")
                    continue
                if _is_error(msg):
                    _, gid, err = msg; raise RuntimeError(f"VAE worker {gid} error: {err}")
                if _is_payload_vae(msg):
                    indices, vae_b = msg
                    vae_writer.write_batch(indices, vae_b)
                    # track rows for verification
                    for g in indices.tolist():
                        sid, off = vae_map.locate(int(g))
                        recent_rows.append((vae_map.paths[sid], off, vae_b.shape[1:]))
                        if len(recent_rows) > 4096: recent_rows.pop(0)
                    gpu_written += len(indices)
                    if (gpu_written % 8192) == 0: vae_writer.flush()
                    pbar.update(len(indices))
            if verifier.error: raise RuntimeError(f"Periodic verification failed: {verifier.error}")

        # tell workers to stop
        for q in inqs: q.put(None)
        # final drain
        while gpu_written < dispatched:
            msg = outq.get()
            if _is_error(msg):
                _, gid, err = msg; raise RuntimeError(f"VAE worker {gid} error: {err}")
            if _is_payload_vae(msg):
                indices, vae_b = msg
                vae_writer.write_batch(indices, vae_b)
                for g in indices.tolist():
                    sid, off = vae_map.locate(int(g))
                    recent_rows.append((vae_map.paths[sid], off, vae_b.shape[1:]))
                    if len(recent_rows) > 4096: recent_rows.pop(0)
                gpu_written += len(indices)
                if (gpu_written % 8192) == 0: vae_writer.flush()
                pbar.update(len(indices))
        if verifier.error: raise RuntimeError(f"Periodic verification failed: {verifier.error}")

    finally:
        pbar.close()
        reporter.stop()
        verifier.stop()
        for p in procs:
            try: p.join(timeout=5)
            except: pass

    vae_writer.flush(); vae_writer.close()
    _save_json(os.path.join(out_dir, 'meta.json'), meta)

    # Final strict FSCK for VAE
    missing = _fsck_group(dataset_root, group="vae", strict=False, jobs=4)
    if sum(len(v) for v in missing.values()) > 0:
        raise RuntimeError(f"[encode-vae] post-run FSCK failed: { {k: len(v) for k,v in missing.items()} }")

    print(f"[vae] wrote {len(paths)} shards to {out_dir} (verified)")


# --------------------------
# Stage 3: DINOv3 from image shards (aligned counts; mapped writes)
# --------------------------

@cli.command(name='encode-dino')
@click.option('--dataset-root', required=True, type=str)
@click.option('--gpus', type=int, default=8, show_default=True)
@click.option('--batch-size', type=int, default=64, show_default=True)
@click.option('--model-name', type=str, default='facebook/dinov3-vit7b16-pretrain-lvd1689m', show_default=True)
@click.option('--dtype', type=click.Choice(['float32','float16','bfloat16']), default='float16', show_default=True)
@click.option('--dino-dim', type=int, default=4096, show_default=True)
@click.option('--patch-size', type=int, default=16, show_default=True)
@click.option('--num-register', type=int, default=4, show_default=True)
def encode_dino(dataset_root, gpus, batch_size, model_name, dtype, dino_dim, patch_size, num_register):
    """Read image shards and write aligned DINOv3 shards: cls, registers, patches."""
    img_meta = _images_meta(dataset_root)
    n = int(img_meta["num_samples"])
    H, W, _ = img_meta["shape"]
    rps = int(img_meta["records_per_shard"])
    cls_shape, regs_shape, patches_shape, _N = _dino_shapes(H, W, dino_dim, patch_size, num_register)
    out_dtype = np.float16 if dtype in ('float16','bfloat16') else np.float32

    out_dir = os.path.join(dataset_root, 'dinov3-vit7b16', 'packed')
    _ensure_dir(out_dir)
    # copy labels/order
    for fname in ('labels.npy','order_keys.txt'):
        src = os.path.join(dataset_root, 'images', 'packed', fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil; shutil.copy(src, dst)

    counts = [int(s["count"]) for s in img_meta["shards"]]
    cls_paths, regs_paths, pat_paths = [], [], []
    cls_w, regs_w, pat_w = [], [], []
    for sid, cnt in enumerate(counts):
        cpath = os.path.join(out_dir, f"dinov3_cls_{sid:05d}.npy")
        rpath = os.path.join(out_dir, f"dinov3_registers_{sid:05d}.npy")
        ppath = os.path.join(out_dir, f"dinov3_patches_{sid:05d}.npy")
        cls_w.append(_open_memmap(cpath, shape=(cnt,*cls_shape), dtype=out_dtype)); cls_paths.append(cpath)
        regs_w.append(_open_memmap(rpath, shape=(cnt,*regs_shape), dtype=out_dtype)); regs_paths.append(rpath)
        pat_w.append(_open_memmap(ppath, shape=(cnt,*patches_shape), dtype=out_dtype)); pat_paths.append(ppath)

    cls_map  = ShardMap(cls_paths,  counts)
    regs_map = ShardMap(regs_paths, counts)
    pat_map  = ShardMap(pat_paths,  counts)

    meta_cls     = {"shape": cls_shape,     "dtype": str(out_dtype), "records_per_shard": int(rps),
                    "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(cls_paths, counts)]}
    meta_regs    = {"shape": regs_shape,    "dtype": str(out_dtype), "records_per_shard": int(rps),
                    "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(regs_paths, counts)]}
    meta_patches = {"shape": patches_shape, "dtype": str(out_dtype), "records_per_shard": int(rps),
                    "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(pat_paths, counts)]}

    # gpu pool
    ctx = mp.get_context('spawn')
    inqs = [ctx.Queue(maxsize=8) for _ in range(gpus)]
    outq = ctx.Queue(maxsize=64)
    procs = []
    for gid in range(gpus):
        p = ctx.Process(target=gpu_worker_dino,
                        args=(gid, inqs[gid], outq, model_name, dtype, dino_dim, patch_size, num_register, H, W))
        p.start(); procs.append(p)

    dispatched = 0
    gpu_written = 0
    workers_ready = set()
    first_batch_seen = set()

    # verification bookkeeping
    recent_cls: List[Tuple[str,int,Tuple[int,...]]] = []
    recent_regs: List[Tuple[str,int,Tuple[int,...]]] = []
    recent_pat: List[Tuple[str,int,Tuple[int,...]]]  = []

    def sample_recent(k):
        random.shuffle(recent_cls); random.shuffle(recent_regs); random.shuffle(recent_pat)
        return {
            "dino.cls":  recent_cls[:k//3],
            "dino.regs": recent_regs[:k//3],
            "dino.pat":  recent_pat[:k//3],
        }

    def check_row(path, row_off, shape):
        mm = np.load(path, mmap_mode='r')
        x = mm[row_off]
        ok = not np.allclose(x, 0.0, atol=0.0)
        del mm
        return ok

    verifier = PeriodicVerifier(sample_fn=sample_recent, check_fn=check_row, interval=20.0, sample_k=192, name="verify-dino")
    verifier.start()

    reporter = PeriodicReporter(
        lambda: dict(dispatched=dispatched, gpu_written=gpu_written, ready=len(workers_ready), first_batches=len(first_batch_seen), total=n),
        interval=10.0, name="dino-progress"
    )
    reporter.start()

    pbar = tqdm(total=n, desc="DINO encode/write")
    rr = 0
    try:
        for idxs, batch in _iter_image_batches(dataset_root, batch_size):
            inqs[rr % gpus].put((idxs, batch))
            rr += 1
            dispatched += len(idxs)
            while True:
                try:
                    msg = outq.get_nowait()
                except queue.Empty:
                    break
                if _is_heartbeat(msg):
                    kind, gid = msg[0], msg[1]
                    if kind == 'ready':
                        workers_ready.add(gid); print(f"[dino] gpu{gid} ready")
                    else:
                        first_batch_seen.add(gid); print(f"[dino] gpu{gid} first batch: idx0={msg[2]} size={msg[3]}")
                    continue
                if _is_error(msg):
                    _, gid, err = msg; raise RuntimeError(f"DINO worker {gid} error: {err}")
                if _is_payload_dino(msg):
                    indices, cls_b, regs_b, pat_b = msg
                    # mapped writes
                    for i, g in enumerate(indices.tolist()):
                        sid, off = cls_map.locate(int(g));  cls_w[sid][off]  = cls_b[i]
                        sid, off = regs_map.locate(int(g)); regs_w[sid][off] = regs_b[i]
                        sid, off = pat_map.locate(int(g));  pat_w[sid][off]  = pat_b[i]
                        # record for verification
                        recent_cls.append((cls_map.paths[sid], off, cls_b.shape[1:]));  recent_cls[:] = recent_cls[-4096:]
                        recent_regs.append((regs_map.paths[sid], off, regs_b.shape[1:])); recent_regs[:] = recent_regs[-4096:]
                        recent_pat.append((pat_map.paths[sid],  off, pat_b.shape[1:]));  recent_pat[:]  = recent_pat[-4096:]
                    gpu_written += len(indices)
                    if (gpu_written % 8192) == 0:
                        for mm in (cls_w + regs_w + pat_w):
                            try: mm.flush()
                            except Exception: pass
                    pbar.update(len(indices))
            if verifier.error: raise RuntimeError(f"Periodic verification failed: {verifier.error}")

        for q in inqs: q.put(None)
        while gpu_written < dispatched:
            msg = outq.get()
            if _is_error(msg):
                _, gid, err = msg; raise RuntimeError(f"DINO worker {gid} error: {err}")
            if _is_payload_dino(msg):
                indices, cls_b, regs_b, pat_b = msg
                for i, g in enumerate(indices.tolist()):
                    sid, off = cls_map.locate(int(g));  cls_w[sid][off]  = cls_b[i]
                    sid, off = regs_map.locate(int(g)); regs_w[sid][off] = regs_b[i]
                    sid, off = pat_map.locate(int(g));  pat_w[sid][off]  = pat_b[i]
                    recent_cls.append((cls_map.paths[sid], off, cls_b.shape[1:]));  recent_cls[:] = recent_cls[-4096:]
                    recent_regs.append((regs_map.paths[sid], off, regs_b.shape[1:])); recent_regs[:] = recent_regs[-4096:]
                    recent_pat.append((pat_map.paths[sid],  off, pat_b.shape[1:]));  recent_pat[:]  = recent_pat[-4096:]
                gpu_written += len(indices)
                if (gpu_written % 8192) == 0:
                    for mm in (cls_w + regs_w + pat_w):
                        try: mm.flush()
                        except Exception: pass
                pbar.update(len(indices))
        if verifier.error: raise RuntimeError(f"Periodic verification failed: {verifier.error}")

    finally:
        pbar.close()
        reporter.stop()
        verifier.stop()
        for p in procs:
            try: p.join(timeout=5)
            except: pass

    # close + meta
    for mm in (cls_w + regs_w + pat_w):
        try: mm.flush()
        except Exception: pass
    for mm in (cls_w + regs_w + pat_w): del mm

    meta = {
        "num_samples": n,
        "cls": meta_cls,
        "registers": meta_regs,
        "patches": meta_patches,
        "dino_dim": int(dino_dim),
        "num_register": int(num_register),
        "patch_size": int(patch_size),
    }
    _save_json(os.path.join(out_dir, 'meta.json'), meta)

    # Final strict FSCK for DINO
    missing = _fsck_group(dataset_root, group="dino", strict=False, jobs=4)
    if sum(len(v) for v in missing.values()) > 0:
        raise RuntimeError(f"[encode-dino] post-run FSCK failed: { {k: len(v) for k,v in missing.items()} }")

    print(f"[dino] wrote {len(cls_paths)} shards to {out_dir} (verified)")


# --------------------------
# Unified: ALL (images + vae + dino) with mapped writes + verification
# --------------------------

@cli.command()
@click.option('--source', required=True, type=str)
@click.option('--dest',   required=True, type=str)
@click.option('--transform', type=click.Choice(['center-crop','center-crop-dhariwal']), default='center-crop-dhariwal', show_default=True)
@click.option('--resolution', required=True, type=str)
@click.option('--shuffle/--no-shuffle', default=True, show_default=True)
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--gpus', type=int, default=8, show_default=True)
@click.option('--batch-size', type=int, default=64, show_default=True)
@click.option('--vae-url', type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--dino-model', type=str, default='facebook/dinov3-vit7b16-pretrain-lvd1689m', show_default=True)
@click.option('--dino-dtype', type=click.Choice(['float32','float16','bfloat16']), default='float16', show_default=True)
@click.option('--dino-dim', type=int, default=4096, show_default=True)
@click.option('--patch-size', type=int, default=16, show_default=True)
@click.option('--num-register', type=int, default=4, show_default=True)
@click.option('--img-shard-bytes',  type=int, default=2_000_000_000, show_default=True)
@click.option('--vae-shard-bytes',  type=int, default=1_000_000_000, show_default=True)
@click.option('--dino-shard-bytes', type=int, default=2_000_000_000, show_default=True)
@click.option('--workers', type=int, default=16, show_default=True, help='CPU decode workers')
def all(source, dest, transform, resolution, shuffle, seed, gpus, batch_size,
        vae_url, dino_model, dino_dtype, dino_dim, patch_size, num_register,
        img_shard_bytes, vae_shard_bytes, dino_shard_bytes, workers):
    """End-to-end: decode -> images write; GPU encodes VAE+DINO; write all out-of-order directly to mapped offsets."""
    W, H = _parse_tuple(resolution)
    w, h = W, H
    refs, rels, _ = _enumerate_images(source)
    n = len(refs)
    if n == 0: raise click.ClickException("No images found.")
    labels = _labels_from_dirs(source if os.path.isdir(source) else os.path.dirname(source), rels)
    order = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(order)

    img_shape = (h, w, 3); img_dtype = np.uint8
    vae_shape = _vae_moments_shape(h, w)  # (8, h/8, w/8)
    vae_dtype = np.float32
    cls_shape, regs_shape, patches_shape, _N = _dino_shapes(h, w, dino_dim, patch_size, num_register)
    dino_out_dtype = np.float16 if dino_dtype in ('float16','bfloat16') else np.float32

    rps_img  = _records_per_shard(img_shard_bytes,     img_shape,     img_dtype)
    rps_vae  = _records_per_shard(vae_shard_bytes,     vae_shape,     vae_dtype)
    rps_dino = min(
        _records_per_shard(dino_shard_bytes, cls_shape,     dino_out_dtype),
        _records_per_shard(dino_shard_bytes, regs_shape,    dino_out_dtype),
        _records_per_shard(dino_shard_bytes, patches_shape, dino_out_dtype),
    )
    rps = min(rps_img, rps_vae, rps_dino)
    shard_count = math.ceil(n / rps)

    # dirs + shared labels/order
    img_dir = os.path.join(dest, 'images', 'packed'); _ensure_dir(img_dir)
    vae_dir = os.path.join(dest, 'vae-sd', 'packed'); _ensure_dir(vae_dir)
    dino_dir= os.path.join(dest, 'dinov3-vit7b16', 'packed'); _ensure_dir(dino_dir)
    np.save(os.path.join(img_dir, 'labels.npy'), labels[order])
    _write_lines(os.path.join(img_dir, 'order_keys.txt'), [f"{i:08d}" for i in order])
    for target in (vae_dir, dino_dir):
        for fname in ('labels.npy','order_keys.txt'):
            import shutil; shutil.copy(os.path.join(img_dir, fname), os.path.join(target, fname))

    # open writers
    img_paths, vae_paths, cls_paths, regs_paths, pat_paths = [], [], [], [], []
    img_w, vae_w, cls_w, regs_w, pat_w = [], [], [], [], []

    for sid in range(shard_count):
        cnt = rps if sid < shard_count-1 else (n - rps*(shard_count-1))

        ipath = os.path.join(img_dir,  f"images_{sid:05d}.npy")
        vpath = os.path.join(vae_dir,  f"vae_latents_{sid:05d}.npy")
        cpath = os.path.join(dino_dir, f"dinov3_cls_{sid:05d}.npy")
        rpath = os.path.join(dino_dir, f"dinov3_registers_{sid:05d}.npy")
        ppath = os.path.join(dino_dir, f"dinov3_patches_{sid:05d}.npy")

        mm_img = _open_memmap(ipath, shape=(cnt, *img_shape), dtype=img_dtype)
        mm_vae = _open_memmap(vpath, shape=(cnt, *vae_shape), dtype=vae_dtype)
        mm_cls = _open_memmap(cpath, shape=(cnt, *cls_shape), dtype=dino_out_dtype)
        mm_regs= _open_memmap(rpath, shape=(cnt, *regs_shape), dtype=dino_out_dtype)
        mm_pat = _open_memmap(ppath, shape=(cnt, *patches_shape), dtype=dino_out_dtype)

        # Early sanity assertions to catch any odd FS behavior immediately
        assert mm_img.shape[0] == cnt and mm_img.shape[1:] == img_shape, f"[init-all] images shard shape mismatch at {ipath}: {mm_img.shape} vs {(cnt, *img_shape)}"
        assert mm_vae.shape[0] == cnt and mm_vae.shape[1:] == vae_shape, f"[init-all] vae shard shape mismatch at {vpath}: {mm_vae.shape} vs {(cnt, *vae_shape)}"
        assert mm_cls.shape[0] == cnt and mm_cls.shape[1:] == cls_shape, f"[init-all] cls shard shape mismatch at {cpath}: {mm_cls.shape} vs {(cnt, *cls_shape)}"
        assert mm_regs.shape[0] == cnt and mm_regs.shape[1:] == regs_shape, f"[init-all] regs shard shape mismatch at {rpath}: {mm_regs.shape} vs {(cnt, *regs_shape)}"
        assert mm_pat.shape[0] == cnt and mm_pat.shape[1:] == patches_shape, f"[init-all] patches shard shape mismatch at {ppath}: {mm_pat.shape} vs {(cnt, *patches_shape)}"

        img_paths.append(ipath);  img_w.append(mm_img)
        vae_paths.append(vpath);  vae_w.append(mm_vae)
        cls_paths.append(cpath);  cls_w.append(mm_cls)
        regs_paths.append(rpath); regs_w.append(mm_regs)
        pat_paths.append(ppath);  pat_w.append(mm_pat)

    # Build counts from the actual writer shapes (source of truth)
    img_counts  = [int(w.shape[0]) for w in img_w]
    vae_counts  = [int(w.shape[0]) for w in vae_w]
    cls_counts  = [int(w.shape[0]) for w in cls_w]
    regs_counts = [int(w.shape[0]) for w in regs_w]
    pat_counts  = [int(w.shape[0]) for w in pat_w]

    # All shards should match; fail fast if not
    if not (img_counts == vae_counts == cls_counts == regs_counts == pat_counts):
        raise RuntimeError(f"[init-all] shard row-count mismatch: "
                        f"img[0]={img_counts[:3]} vae[0]={vae_counts[:3]} "
                        f"cls[0]={cls_counts[:3]} regs[0]={regs_counts[:3]} pat[0]={pat_counts[:3]}")

    img_map  = ShardMap(img_paths,  img_counts)
    vae_map  = ShardMap(vae_paths,  vae_counts)
    cls_map  = ShardMap(cls_paths,  cls_counts)
    regs_map = ShardMap(regs_paths, regs_counts)
    pat_map  = ShardMap(pat_paths,  pat_counts)

    print(f"[init-all] first shard shapes: "
        f"img{img_w[0].shape} vae{vae_w[0].shape} cls{cls_w[0].shape} regs{regs_w[0].shape} pat{pat_w[0].shape}",
        flush=True)

    # metas
    img_meta = {"num_samples": int(n), "shape": img_shape, "dtype": str(img_dtype), "records_per_shard": int(rps),
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(img_paths, img_counts)]}
    vae_meta = {"num_samples": int(n), "shape": vae_shape, "dtype": str(vae_dtype), "records_per_shard": int(rps),
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(vae_paths, img_counts)]}
    cls_meta = {"shape": cls_shape, "dtype": str(dino_out_dtype), "records_per_shard": int(rps),
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(cls_paths, img_counts)]}
    regs_meta= {"shape": regs_shape,"dtype": str(dino_out_dtype), "records_per_shard": int(rps),
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(regs_paths, img_counts)]}
    pat_meta = {"shape": patches_shape,"dtype": str(dino_out_dtype), "records_per_shard": int(rps),
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(pat_paths, img_counts)]}

    img_writer = ShardedWriter(img_map, img_w)
    vae_writer = ShardedWriter(vae_map, vae_w)
    # DINO writers we access directly due to three streams but mapping uses ShardMap

    # GPU pool
    ctx = mp.get_context('spawn')
    inqs = [ctx.Queue(maxsize=8) for _ in range(gpus)]
    outq = ctx.Queue(maxsize=64)
    procs = []
    for gid in range(gpus):
        p = ctx.Process(target=gpu_worker_all,
                        args=(gid, inqs[gid], outq,
                              vae_url, dino_model, dino_dtype,
                              dino_dim, patch_size, num_register, h, w))
        p.start(); procs.append(p)

    # counters + verifier
    decoded_written = 0
    gpu_dispatched = 0
    gpu_written = 0
    workers_ready = set()
    first_batch_seen = set()

    recent_vae: List[Tuple[str,int,Tuple[int,...]]] = []
    recent_cls: List[Tuple[str,int,Tuple[int,...]]] = []
    recent_regs: List[Tuple[str,int,Tuple[int,...]]] = []
    recent_pat: List[Tuple[str,int,Tuple[int,...]]] = []

    def sample_recent(k):
        random.shuffle(recent_vae); random.shuffle(recent_cls); random.shuffle(recent_regs); random.shuffle(recent_pat)
        return {
            "vae":       recent_vae[:k//4],
            "dino.cls":  recent_cls[:k//4],
            "dino.regs": recent_regs[:k//4],
            "dino.pat":  recent_pat[:k//4],
        }

    def check_row(path, row_off, shape):
        mm = np.load(path, mmap_mode='r')
        x = mm[row_off]
        ok = not np.allclose(x, 0.0, atol=0.0)
        del mm
        return ok

    verifier = PeriodicVerifier(sample_fn=sample_recent, check_fn=check_row, interval=20.0, sample_k=256, name="verify-all")
    verifier.start()

    reporter = PeriodicReporter(
        lambda: dict(decoded=decoded_written, dispatched=gpu_dispatched, gpu_written=gpu_written,
                     ready=len(workers_ready), first_batches=len(first_batch_seen), total=n),
        interval=10.0, name="all-progress"
    )
    reporter.start()

    def _flush_to_gpu(rr_counter: List[int]):
        nonlocal gpu_dispatched, staged_indices, staged_batch
        if not staged_indices: return
        inds = np.asarray(staged_indices, dtype=np.int64)
        arr  = np.stack(staged_batch, axis=0)
        inqs[rr_counter[0] % gpus].put((inds, arr))
        rr_counter[0] += 1
        gpu_dispatched += len(inds)
        staged_indices = []
        staged_batch = []

    def _drain_outq(nonblock=True):
        nonlocal gpu_written
        wrote = 0
        while True:
            try:
                msg = outq.get_nowait() if nonblock else outq.get()
            except queue.Empty:
                break
            if _is_heartbeat(msg):
                kind, gid = msg[0], msg[1]
                if kind == 'ready':
                    workers_ready.add(gid)
                    print(f"[all] gpu{gid} ready ({msg[2]})")
                else:
                    first_batch_seen.add(gid)
                    print(f"[all] gpu{gid} first batch: idx0={msg[2]} size={msg[3]}")
                continue
            if _is_error(msg):
                _, gid, err = msg
                raise RuntimeError(f"GPU worker {gid} error: {err}")
            if _is_payload_all(msg):
                indices, vae_b, cls_b, regs_b, pat_b = msg
                # write VAE
                vae_writer.write_batch(indices, vae_b)
                for g in indices.tolist():
                    sid, off = vae_map.locate(int(g))
                    recent_vae.append((vae_map.paths[sid], off, vae_b.shape[1:])); recent_vae[:] = recent_vae[-4096:]
                # write DINO
                for i, g in enumerate(indices.tolist()):
                    sid, off = cls_map.locate(int(g));  cls_w[sid][off]  = cls_b[i]
                    sid, off = regs_map.locate(int(g)); regs_w[sid][off] = regs_b[i]
                    sid, off = pat_map.locate(int(g));  pat_w[sid][off]  = pat_b[i]
                    recent_cls.append((cls_map.paths[sid], off, cls_b.shape[1:]));  recent_cls[:] = recent_cls[-4096:]
                    recent_regs.append((regs_map.paths[sid], off, regs_b.shape[1:])); recent_regs[:] = recent_regs[-4096:]
                    recent_pat.append((pat_map.paths[sid],  off, pat_b.shape[1:]));  recent_pat[:]  = recent_pat[-4096:]
                wrote += len(indices)
        if wrote:
            gpu_written += wrote
            if (gpu_written % 8192) == 0:
                img_writer.flush(); vae_writer.flush()
                for mm in (cls_w + regs_w + pat_w):
                    try: mm.flush()
                    except Exception: pass
            p_gpu.update(wrote)
        if verifier.error: raise RuntimeError(f"Periodic verification failed: {verifier.error}")

    # progress trackers
    p_decode = tqdm(total=n, desc="Decode/write images")
    p_gpu    = tqdm(total=n, desc="VAE+DINO write")

    # launch decode workers
    rr_box = [0]  # mutable int
    staged_indices: List[int] = []
    staged_batch: List[np.ndarray] = []
    with _start_image_pool(workers) as pool:
        try:
            CHUNK = 4096
            for i0 in range(0, n, CHUNK):
                batch = [(i, refs[order[i]], w, h, transform) for i in range(i0, min(n, i0+CHUNK))]
                for idx, arr in pool.imap_unordered(_img_worker, batch, chunksize=64):
                    # write image immediately (mapped)
                    img_writer.write_batch(np.asarray([idx], dtype=np.int64), np.asarray([arr], dtype=np.uint8))
                    decoded_written += 1
                    if (decoded_written % 8192) == 0: img_writer.flush()
                    p_decode.update(1)
                    # stage for GPU
                    staged_indices.append(idx)
                    staged_batch.append(arr)
                    if len(staged_indices) >= batch_size:
                        _flush_to_gpu(rr_box)
                        _drain_outq(nonblock=True)  # keep outq flowing
            # flush leftovers
            if staged_indices:
                _flush_to_gpu(rr_box)
        finally:
            p_decode.close()

    # stop workers; final drain until all dispatched rows are written
    for q in inqs: q.put(None)

    # Keep draining until we've written everything we dispatched.
    while gpu_written < gpu_dispatched:
        try:
            msg = outq.get(timeout=5.0)
        except queue.Empty:
            # If nothing arrives for a bit, check if any worker is still alive.
            # If all workers are gone and we're still short, bail with a clear error.
            all_dead = all(not p.is_alive() for p in procs)
            if all_dead and gpu_written < gpu_dispatched:
                raise RuntimeError(
                    f"[all] workers exited early: gpu_written={gpu_written} < dispatched={gpu_dispatched}"
                )
            continue

        if _is_heartbeat(msg):
            # ignore heartbeats here
            continue
        if _is_error(msg):
            _, gid, err = msg
            raise RuntimeError(f"GPU worker {gid} error: {err}")
        if _is_payload_all(msg):
            indices, vae_b, cls_b, regs_b, pat_b = msg

            # VAE
            vae_writer.write_batch(indices, vae_b)
            for g in indices.tolist():
                sid, off = vae_map.locate(int(g))
                recent_vae.append((vae_map.paths[sid], off, vae_b.shape[1:])); recent_vae[:] = recent_vae[-4096:]

            # DINO
            for i, g in enumerate(indices.tolist()):
                sid, off = cls_map.locate(int(g));  cls_w[sid][off]  = cls_b[i]
                sid, off = regs_map.locate(int(g)); regs_w[sid][off] = regs_b[i]
                sid, off = pat_map.locate(int(g));  pat_w[sid][off]  = pat_b[i]
                recent_cls.append((cls_map.paths[sid], off, cls_b.shape[1:]));  recent_cls[:] = recent_cls[-4096:]
                recent_regs.append((regs_map.paths[sid], off, regs_b.shape[1:])); recent_regs[:] = recent_regs[-4096:]
                recent_pat.append((pat_map.paths[sid],  off, pat_b.shape[1:]));  recent_pat[:]  = recent_pat[-4096:]

            gpu_written += len(indices)
            if (gpu_written % 8192) == 0:
                img_writer.flush(); vae_writer.flush()
                for mm in (cls_w + regs_w + pat_w):
                    try: mm.flush()
                    except Exception: pass
            p_gpu.update(len(indices))

    if verifier.error:
        raise RuntimeError(f"Periodic verification failed: {verifier.error}")

    p_gpu.close()
    reporter.stop()
    verifier.stop()


    # metas & flush
    _save_json(os.path.join(img_dir, 'meta.json'), img_meta)
    _save_json(os.path.join(vae_dir, 'meta.json'), vae_meta)
    _save_json(os.path.join(dino_dir, 'meta.json'),
               {"num_samples": int(n), "cls": cls_meta, "registers": regs_meta, "patches": pat_meta,
                "dino_dim": int(dino_dim), "num_register": int(num_register), "patch_size": int(patch_size)})
    img_writer.flush(); vae_writer.flush()
    for mm in (cls_w + regs_w + pat_w):
        try: mm.flush()
        except Exception: pass
    img_writer.close(); vae_writer.close()
    for mm in (cls_w + regs_w + pat_w): del mm
    for p in procs:
        try: p.join(timeout=5)
        except: pass

    # Final strict FSCK for all streams
    miss_all = _fsck_group(dest, group="all", strict=False, jobs=4)
    if sum(len(v) for v in miss_all.values()) > 0:
        raise RuntimeError(f"[all] post-run FSCK failed: { {k: len(v) for k,v in miss_all.items()} }")

    print(f"[all] shards — images:{len(img_meta['shards'])} vae:{len(vae_meta['shards'])} dino:{len(cls_meta['shards'])} (verified)")


# ======================================================================
#                           FSCK + STATS (shared)
# ======================================================================

def _discover_group_specs(dataset_root: str):
    """Return dicts describing shards for images/vae/dino by scanning filenames; header-only I/O."""
    def scan(prefix, subdir):
        root = os.path.join(dataset_root, subdir, 'packed')
        if not os.path.isdir(root): return None
        files = sorted([f for f in os.listdir(root) if f.startswith(prefix) and f.endswith('.npy')])
        if not files: return None
        paths = [os.path.join(root, f) for f in files]
        counts, shapes, dtypes = [], None, None
        for p in paths:
            mm = np.load(p, mmap_mode='r')
            if shapes is None:
                shapes = mm.shape[1:]
                dtypes = str(mm.dtype)
            counts.append(mm.shape[0])
            del mm
        return {
            "root": root,
            "paths": paths,
            "counts": counts,
            "sample_shape": tuple(shapes),
            "dtype": dtypes,
            "total": int(sum(counts)),
        }

    img = scan('images_', 'images')
    vae = scan('vae_latents_', 'vae-sd')
    # DINO: three streams
    dino_root = os.path.join(dataset_root, 'dinov3-vit7b16', 'packed')
    if os.path.isdir(dino_root):
        def scan_one(prefix):
            files = sorted([f for f in os.listdir(dino_root) if f.startswith(prefix) and f.endswith('.npy')])
            if not files: return None
            paths = [os.path.join(dino_root, f) for f in files]
            counts, shapes, dtypes = [], None, None
            for p in paths:
                mm = np.load(p, mmap_mode='r')
                if shapes is None:
                    shapes = mm.shape[1:]
                    dtypes = str(mm.dtype)
                counts.append(mm.shape[0]); del mm
            return {"paths": paths, "counts": counts, "sample_shape": tuple(shapes), "dtype": dtypes}
        dino = {
            "root": dino_root,
            "cls":  scan_one('dinov3_cls_'),
            "regs": scan_one('dinov3_registers_'),
            "pat":  scan_one('dinov3_patches_'),
        }
        if dino["cls"] and dino["regs"] and dino["pat"]:
            dino["total"] = int(sum(dino["cls"]["counts"]))
        else:
            dino = None
    else:
        dino = None
    return {"images": img, "vae": vae, "dino": dino}

def _global_offsets(counts: List[int]) -> List[int]:
    offs, acc = [], 0
    for c in counts:
        offs.append(acc); acc += c
    return offs

def _fast_missing_mask(view_2d: np.ndarray) -> np.ndarray:
    if view_2d.shape[1] <= 32:
        head_all_zero = (view_2d == 0).all(axis=1)
        return head_all_zero
    head = view_2d[:, :16]
    tail = view_2d[:, -16:]
    return (head == 0).all(axis=1) & (tail == 0).all(axis=1)

def _strict_missing_mask(view_2d: np.ndarray) -> np.ndarray:
    return (view_2d == 0).all(axis=1)

def _scan_npy_missing_rows(path: str, fast: bool = True) -> np.ndarray:
    mm = np.load(path, mmap_mode='r')
    view = np.reshape(mm, (mm.shape[0], -1))
    mask = _fast_missing_mask(view) if fast else _strict_missing_mask(view)
    missing = np.nonzero(mask)[0].astype(np.int64)
    del mm
    return missing

def _tqdm_map(fn, iterable, total, desc, jobs: int = 0):
    if jobs and jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out = []
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(fn, item) for item in iterable]
            for _ in tqdm(as_completed(futs), total=total, desc=desc):
                pass
            for f in futs: out.append(f.result())
        return out
    else:
        out = []
        for item in tqdm(iterable, total=total, desc=desc):
            out.append(fn(item))
        return out

def _fsck_group(dataset_root: str, group: str, strict: bool, jobs: int) -> Dict[str, List[int]]:
    specs = _discover_group_specs(dataset_root)
    missing: Dict[str, List[int]] = {}

    def scan_stream(name, spec):
        if not spec: return
        offs = _global_offsets(spec["counts"])
        def per_shard(i):
            p = spec["paths"][i]
            row_off = offs[i]
            miss_local = _scan_npy_missing_rows(p, fast=(not strict))
            return (p, row_off, miss_local)
        results = _tqdm_map(per_shard, range(len(spec["paths"])), total=len(spec["paths"]), desc=f"scan {name}", jobs=jobs)
        miss_global = []
        for p, row_off, miss_local in results:
            miss_global.extend((row_off + miss_local).tolist())
        missing[name] = sorted(miss_global)

    if group in ('images','all'): scan_stream('images', specs["images"])
    if group in ('vae','all'):    scan_stream('vae',    specs["vae"])
    if group in ('dino','all') and specs["dino"]:
        scan_stream('dino.cls',  specs["dino"]["cls"])
        scan_stream('dino.regs', specs["dino"]["regs"])
        scan_stream('dino.pat',  specs["dino"]["pat"])
    return missing


@cli.command(name='fsck')
@click.option('--dataset-root', required=True, type=str)
@click.option('--group', type=click.Choice(['images','vae','dino','all']), default='all', show_default=True)
@click.option('--strict/--fast', default=False, show_default=True, help="Strict = read full rows. Fast = peek bytes (default).")
@click.option('--jobs', type=int, default=0, help="Parallel shard header/row scanning (threads). 0=serial.")
def fsck(dataset_root, group, strict, jobs):
    """Verify which rows look unwritten (all-zeros) for each shard stream."""
    missing = _fsck_group(dataset_root, group, strict, jobs)
    for k,v in missing.items():
        print(f"[fsck] {k}: missing {len(v)} rows")
    if not missing:
        print("[fsck] nothing to scan (group not found).")


@cli.command(name="stats")
@click.option('--dataset-root', required=True, type=str)
def stats(dataset_root):
    S = _discover_group_specs(dataset_root)
    def pr(name, spec):
        if not spec:
            print(f"{name}: (missing)")
            return
        if "paths" in spec and name != "dino":
            print(f"{name}: shards={len(spec['paths'])} total={sum(spec['counts'])} "
                  f"sample_shape={spec['sample_shape']} dtype={spec['dtype']} "
                  f"first={spec['counts'][0]} last={spec['counts'][-1]}")
        else:
            print(f"{name}: {(spec and spec.get('total')) or 0}")
            if spec:
                for k in ("cls","regs","pat"):
                    sub = spec[k]
                    print(f"  {k}: shards={len(sub['paths'])} first={sub['counts'][0]} last={sub['counts'][-1]} "
                          f"shape={sub['sample_shape']} dtype={sub['dtype']}")
    pr("images", S["images"])
    pr("vae",    S["vae"])
    pr("dino",   S["dino"])

@cli.command(name="repair")
@click.option('--dataset-root', required=True, type=str)
@click.option('--which', type=click.Choice(['images','vae','dino','all']), default='all', show_default=True)
@click.option('--gpus', type=int, default=8, show_default=True)
@click.option('--batch-size', type=int, default=64, show_default=True)
# Needed only if images need repair or if VAE/DINO need rows whose images are missing.
@click.option('--source', type=str, default=None, help="Original image folder or .zip (required if images need repair)")
@click.option('--resolution', type=str, default=None, help="WxH used during build (required if images need repair)")
@click.option('--transform', type=click.Choice(['center-crop','center-crop-dhariwal']), default='center-crop-dhariwal', show_default=True)
@click.option('--dino-model', type=str, default='facebook/dinov3-vit7b16-pretrain-lvd1689m', show_default=True)
@click.option('--dino-dtype', type=click.Choice(['float32','float16','bfloat16']), default='float16', show_default=True)
@click.option('--dino-dim', type=int, default=4096, show_default=True)
@click.option('--patch-size', type=int, default=16, show_default=True)
@click.option('--num-register', type=int, default=4, show_default=True)
@click.option('--vae-url', type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
def repair(dataset_root, which, gpus, batch_size,
           source, resolution, transform,
           dino_model, dino_dtype, dino_dim, patch_size, num_register, vae_url):
    """
    Fill any all-zero rows in images / vae / dino shards.
    - Rebuilds meta.json for repaired groups.
    - For images: decodes from --source using --resolution/--transform.
    - For vae/dino: encodes from packed images when present; falls back to --source for rows whose images were missing.
    """
    def _guess_rps(counts: List[int]) -> int:
        if not counts: return 0
        try:
            from collections import Counter
            return Counter(counts).most_common(1)[0][0]
        except Exception:
            return max(counts)

    # 0) Discover current shard specs
    specs = _discover_group_specs(dataset_root)

    # 1) Find missing rows (strict)
    scan_group = 'all' if which == 'all' else which
    missing = _fsck_group(dataset_root, group=scan_group, strict=False, jobs=4)

    miss_images = set(missing.get('images', []))
    miss_vae    = set(missing.get('vae', []))
    miss_cls    = set(missing.get('dino.cls', []))
    miss_regs   = set(missing.get('dino.regs', []))
    miss_pat    = set(missing.get('dino.pat', []))
    miss_dino   = set().union(miss_cls, miss_regs, miss_pat)

    need_images = (which in ('images','all')) and len(miss_images) > 0
    need_vae    = (which in ('vae','all'))    and len(miss_vae) > 0
    need_dino   = (which in ('dino','all'))   and len(miss_dino) > 0

    if not (need_images or need_vae or need_dino):
        # Even if nothing missing, refresh meta.json if user asked a subset; keep it simple and safe.
        print("[repair] nothing missing; rewriting meta.json files for consistency.")
        _rewrite_meta_files(dataset_root)
        return

    # 2) Load image meta & open image shards (read/write for image repair; read-only otherwise)
    img_root = os.path.join(dataset_root, 'images', 'packed')
    if not os.path.isdir(img_root):
        raise click.ClickException(f"[repair] images shard dir missing: {img_root} (cannot proceed)")

    # Build maps from existing images meta or scan headers
    if os.path.isfile(os.path.join(img_root, 'meta.json')):
        img_meta = _load_json(os.path.join(img_root, 'meta.json'))
        img_paths  = [os.path.join(img_root, s["file"]) for s in img_meta["shards"]]
        img_counts = [int(s["count"]) for s in img_meta["shards"]]
        H, W, _ = img_meta["shape"]
    else:
        # Reconstruct from files
        files = sorted([f for f in os.listdir(img_root) if f.startswith("images_") and f.endswith(".npy")])
        if not files:
            raise click.ClickException(f"[repair] no images shards under {img_root}")
        img_paths, img_counts, shape0 = [], [], None
        for f in files:
            p = os.path.join(img_root, f)
            mm = np.load(p, mmap_mode='r'); shp = mm.shape; del mm
            if shape0 is None: shape0 = shp[1:]
            img_paths.append(p); img_counts.append(int(shp[0]))
        H, W, _ = shape0
        img_meta = {"num_samples": int(sum(img_counts)), "shape": [H, W, 3], "dtype": "uint8",
                    "records_per_shard": _guess_rps(img_counts),
                    "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(img_paths, img_counts)]}
        _save_json(os.path.join(img_root, 'meta.json'), img_meta)

    img_map = ShardMap(img_paths, img_counts)

    # Read order mapping to reconstruct sources if needed
    order_keys_path = os.path.join(img_root, 'order_keys.txt')
    if need_images or need_vae or need_dino:
        if not os.path.isfile(order_keys_path):
            if need_images:
                raise click.ClickException("[repair] order_keys.txt missing; cannot map global indices to source files for images repair.")
        if os.path.isfile(order_keys_path):
            with open(order_keys_path, 'r') as f:
                order = np.asarray([int(line.strip()) for line in f], dtype=np.int64)
            if order.shape[0] != img_meta["num_samples"]:
                print(f"[repair][warn] order length {order.shape[0]} != num_samples {img_meta['num_samples']}")

    # 3) If any row needs image content from source, set up decode source
    will_decode_from_source = need_images or ( (need_vae or need_dino) and any(g in miss_images for g in (miss_vae | miss_dino)) )
    refs = None
    if will_decode_from_source:
        if source is None or resolution is None:
            raise click.ClickException("--source and --resolution are required because some image rows must be decoded.")
        Wsrc, Hsrc = _parse_tuple(resolution)
        if (Hsrc, Wsrc) != (H, W):
            print(f"[repair][warn] passed resolution {Wsrc}x{Hsrc} differs from images meta {W}x{H}. Using {Wsrc}x{Hsrc} for decode and writing into the existing shard shape {H}x{W}.")
            # For safety we’ll decode to (Wsrc,Hsrc) then resize to (W,H) with the same transform function.
        refs, rels, _ = _enumerate_images(source)
        if len(rels) < order.shape[0]:
            raise click.ClickException(f"[repair] source contains {len(rels)} files but dataset expects at least {order.shape[0]} (from order_keys.txt).")

    # 4) Open images shards in r+ if repairing images
    img_writers = []
    for p in img_paths:
        mm = np.lib.format.open_memmap(p, mode=('r+' if need_images else 'r'), dtype=np.uint8, shape=np.load(p, mmap_mode='r').shape)
        img_writers.append(mm)

    # helpers for images IO
    def _write_image_row(gidx: int, arr: np.ndarray):
        sid, off = img_map.locate(int(gidx))
        img_writers[sid][off] = arr

    def _read_image_row(gidx: int) -> Optional[np.ndarray]:
        sid, off = img_map.locate(int(gidx))
        return img_writers[sid][off]

    # decode helper from source (uses order mapping)
    def _decode_from_source(gidx: int) -> np.ndarray:
        # global -> original ref index via order[]
        ref_idx = int(order[gidx])
        ref = refs[ref_idx]
        # do the same transform pipeline as build
        img = _open_pil(ref)
        arr = np.array(img, dtype=np.uint8)
        if transform is None:
            out = _scale(arr, W, H)
        elif transform == 'center-crop':
            out = _center_crop(arr, W, H)
        elif transform == 'center-crop-dhariwal':
            assert W == H, "Dhariwal crop requires square WxH"
            out = _center_crop_dhariwal(arr, W)
        else:
            raise RuntimeError(f"Unknown transform {transform}")
        # Ensure exact target size
        if out.shape[:2] != (H, W):
            out = _scale(out, W, H)
        return out

    # 5) Open / map VAE writers if needed
    vae_dir = os.path.join(dataset_root, 'vae-sd', 'packed')
    vae_writers, vae_paths = [], []
    vae_map = None
    vae_shape = None
    if need_vae or os.path.isdir(vae_dir):
        if not os.path.isdir(vae_dir):
            raise click.ClickException(f"[repair] VAE dir missing: {vae_dir}")
        files = sorted([f for f in os.listdir(vae_dir) if f.startswith("vae_latents_") and f.endswith(".npy")])
        for f in files:
            p = os.path.join(vae_dir, f)
            hdr = np.load(p, mmap_mode='r'); shp, dt = hdr.shape, hdr.dtype; del hdr
            if vae_shape is None: vae_shape = shp[1:]
            vae_writers.append(np.lib.format.open_memmap(p, mode='r+', dtype=dt, shape=shp))
            vae_paths.append(p)
        vae_counts = [int(w.shape[0]) for w in vae_writers]
        vae_map = ShardMap(vae_paths, vae_counts)

    # 6) Open / map DINO writers if needed
    dino_dir = os.path.join(dataset_root, 'dinov3-vit7b16', 'packed')
    cls_w, regs_w, pat_w = [], [], []
    cls_paths, regs_paths, pat_paths = [], [], []
    cls_map = regs_map = pat_map = None
    cls_shape = regs_shape = pat_shape = None
    dino_dtype_np = np.float16 if dino_dtype in ('float16','bfloat16') else np.float32

    if need_dino or os.path.isdir(dino_dir):
        if not os.path.isdir(dino_dir):
            raise click.ClickException(f"[repair] DINO dir missing: {dino_dir}")
        def _open_rw(prefix, out_list, path_list):
            files = sorted([f for f in os.listdir(dino_dir) if f.startswith(prefix) and f.endswith(".npy")])
            for f in files:
                p = os.path.join(dino_dir, f)
                hdr = np.load(p, mmap_mode='r'); shp, dt = hdr.shape, hdr.dtype; del hdr
                out_list.append(np.lib.format.open_memmap(p, mode='r+', dtype=dt, shape=shp))
                path_list.append(p)
        _open_rw('dinov3_cls', cls_w, cls_paths)
        _open_rw('dinov3_registers', regs_w, regs_paths)
        _open_rw('dinov3_patches', pat_w, pat_paths)
        if cls_w:
            cls_shape = cls_w[0].shape[1:]
            regs_shape = regs_w[0].shape[1:]
            pat_shape  = pat_w[0].shape[1:]
            cls_map  = ShardMap(cls_paths,  [int(w.shape[0]) for w in cls_w])
            regs_map = ShardMap(regs_paths, [int(w.shape[0]) for w in regs_w])
            pat_map  = ShardMap(pat_paths,  [int(w.shape[0]) for w in pat_w])

    # 7) If repairing images, do that first (so VAE/DINO can read from images)
    if need_images:
        from tqdm import tqdm as _tqdm
        for g in _tqdm(sorted(miss_images), desc="[repair] images"):
            arr = _decode_from_source(g)
            _write_image_row(g, arr)

    # 8) Build the set of indices that need VAE/DINO writes
    todo_for_encode = sorted((miss_vae if need_vae else set()) | (miss_dino if need_dino else set()))
    if todo_for_encode:
        # GPU workers
        ctx = mp.get_context('spawn')
        inqs = [ctx.Queue(maxsize=8) for _ in range(gpus)]
        outq = ctx.Queue(maxsize=64)
        procs = []
        for gid in range(gpus):
            p = ctx.Process(target=gpu_worker_all,
                            args=(gid, inqs[gid], outq, vae_url,
                                  dino_model, dino_dtype, dino_dim, patch_size, num_register, H, W))
            p.start(); procs.append(p)

        # dispatch helper: assemble batches from images memmaps; if image row is zero/missing, decode from source
        dispatched = written = 0
        buf_idx, buf_img = [], []

        def _flush(rr):
            nonlocal dispatched, buf_idx, buf_img
            if not buf_idx: return rr
            inqs[rr % gpus].put((np.asarray(buf_idx, np.int64), np.stack(buf_img, axis=0)))
            dispatched += len(buf_idx)
            buf_idx, buf_img = [], []
            return rr + 1

        rr = 0
        from tqdm import tqdm as _tqdm
        miss_images_set = miss_images  # already a set

        for g in _tqdm(todo_for_encode, desc="[repair] encode (vae/dino)"):
            if g in miss_images_set:
                if not will_decode_from_source:
                    raise click.ClickException("[repair] need to encode vae/dino but corresponding image row is missing and --source not provided.")
                arr = _decode_from_source(g)
            else:
                arr = _read_image_row(g)
            buf_idx.append(int(g)); buf_img.append(arr)
            if len(buf_idx) >= batch_size:
                rr = _flush(rr)

        if buf_idx:
            rr = _flush(rr)

        for q in inqs: q.put(None)

        # Drain until all written
        while written < dispatched:
            try:
                msg = outq.get(timeout=10.0)
            except queue.Empty:
                # if no progress and all workers dead, error out
                if all(not p.is_alive() for p in procs) and written < dispatched:
                    raise RuntimeError(f"[repair] workers exited early: written={written} < dispatched={dispatched}")
                continue

            if _is_heartbeat(msg):
                continue
            if _is_error(msg):
                _, gid, err = msg; raise RuntimeError(f"[repair] worker {gid} error: {err}")
            if _is_payload_all(msg):
                indices, vae_b, cls_b, regs_b, pat_b = msg
                for i, g in enumerate(indices.tolist()):
                    if need_vae and vae_writers:
                        sid, off = vae_map.locate(int(g)); vae_writers[sid][off] = vae_b[i]
                    if need_dino and cls_w:
                        sid, off = cls_map.locate(int(g));  cls_w[sid][off]  = cls_b[i]
                        sid, off = regs_map.locate(int(g)); regs_w[sid][off] = regs_b[i]
                        sid, off = pat_map.locate(int(g));  pat_w[sid][off]  = pat_b[i]
                written += len(indices)

        # join workers
        for p in procs:
            try: p.join(timeout=5)
            except: pass

    # 9) Flush and close
    for mm in img_writers + vae_writers + cls_w + regs_w + pat_w:
        try: mm.flush()
        except: pass
    for mm in img_writers + vae_writers + cls_w + regs_w + pat_w:
        try: del mm
        except: pass

    # 10) Rebuild meta.json files for repaired groups (and generally keep metas consistent)
    _rewrite_meta_files(dataset_root)

    # 11) Verify again
    post = _fsck_group(dataset_root, group=scan_group, strict=False, jobs=4)
    summary = {k: len(v) for k,v in post.items()}
    print(f"[repair] done. Remaining missing rows per stream: {summary}")


def _rewrite_meta_files(dataset_root: str):
    """
    Re-scan shards on disk and rewrite meta.json files for images / vae / dino.
    Does not change data; metadata only.
    """
    specs = _discover_group_specs(dataset_root)

    # images meta
    if specs["images"]:
        root = specs["images"]["root"]
        paths = [os.path.join(root, os.path.basename(p)) for p in specs["images"]["paths"]]
        counts = specs["images"]["counts"]
        shp = list(specs["images"]["sample_shape"])
        dtype = specs["images"]["dtype"]
        meta = {
            "num_samples": int(sum(counts)),
            "shape": shp,
            "dtype": dtype,
            "records_per_shard": max(counts) if counts else 0,
            "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(paths, counts)],
        }
        _save_json(os.path.join(root, 'meta.json'), meta)

    # vae meta
    if specs["vae"]:
        root = specs["vae"]["root"]
        paths = [os.path.join(root, os.path.basename(p)) for p in specs["vae"]["paths"]]
        counts = specs["vae"]["counts"]
        shp = list(specs["vae"]["sample_shape"])
        dtype = specs["vae"]["dtype"]
        meta = {
            "num_samples": int(sum(counts)),
            "shape": shp,
            "dtype": dtype,
            "records_per_shard": max(counts) if counts else 0,
            "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(paths, counts)],
        }
        _save_json(os.path.join(root, 'meta.json'), meta)

    # dino meta
    if specs["dino"]:
        root = specs["dino"]["root"]
        def m(sub):
            return {
                "shape": list(sub["sample_shape"]),
                "dtype": sub["dtype"],
                "records_per_shard": max(sub["counts"]) if sub["counts"] else 0,
                "shards": [{"file": os.path.basename(p), "count": int(c)} for p,c in zip(sub["paths"], sub["counts"])],
            }
        meta = {
            "num_samples": int(specs["dino"]["total"]),
            "cls":  m(specs["dino"]["cls"]),
            "registers": m(specs["dino"]["regs"]),
            "patches":   m(specs["dino"]["pat"]),
            # Optional hints — try to infer from shapes; keep previous fields if present
        }
        _save_json(os.path.join(root, 'meta.json'), meta)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    cli()
