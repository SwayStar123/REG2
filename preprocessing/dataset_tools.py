# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from dataclasses import dataclass
import io
import json
import multiprocessing as mp
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

# NOTE: Heavy encoder imports are deferred to worker functions so that lightweight
# utilities (e.g., pack_npy_directory) can be used without full model deps.

#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------



#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

def scale_image(width, height, img):
    """Scale image to specified dimensions."""
    w = img.shape[1]
    h = img.shape[0]
    if width == w and height == h:
        return img
    img = PIL.Image.fromarray(img, 'RGB')
    ww = width if width is not None else w
    hh = height if height is not None else h
    img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_image(width, height, img):
    """Center crop and resize image."""
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    return np.array(img)

def center_crop_wide_image(width, height, img):
    """Center crop wide image."""
    ch = int(np.round(width * img.shape[0] / img.shape[1]))
    if img.shape[1] < width or ch < height:
        return None

    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    img = np.array(img)

    canvas = np.zeros([width, width, 3], dtype=np.uint8)
    canvas[(width - height) // 2 : (width + height) // 2, :] = img
    return canvas

def center_crop_imagenet_image(image_size, arr):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = PIL.Image.fromarray(arr)
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

def transform_image_worker(args):
    """Worker function for parallel image transformation."""
    idx, image_data, transform_type, output_width, output_height = args
    try:
        img = image_data.img
        
        if transform_type is None:
            img = scale_image(output_width, output_height, img)
        elif transform_type == 'center-crop':
            img = center_crop_image(output_width, output_height, img)
        elif transform_type == 'center-crop-wide':
            img = center_crop_wide_image(output_width, output_height, img)
        elif transform_type == 'center-crop-dhariwal':
            img = center_crop_imagenet_image(output_width, img)
        else:
            raise ValueError(f'Unknown transform type: {transform_type}')
            
        if img is None:
            return None
        return idx, img, image_data.label
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return None

def encode_image_worker(args):
    """Worker function for parallel VAE encoding."""
    gpu_id, batch_data, model_url = args
    device = torch.device(f'cuda:{gpu_id}')
    # Lazy import to avoid requiring encoders for unrelated utilities
    try:
        from encoders import StabilityVAEEncoder
    except ModuleNotFoundError as e:
        raise RuntimeError("StabilityVAEEncoder not available. Ensure PYTHONPATH includes project root or install encoders module.") from e
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=1)
    results = []
    
    for idx, image_data in batch_data:
        try:
            img_tensor = torch.tensor(image_data.img).to(device).permute(2, 0, 1).unsqueeze(0)
            mean_std = vae.encode_pixels(img_tensor)[0].cpu().numpy()
            results.append((idx, mean_std, image_data.label))
        except Exception as e:
            print(f"Error encoding image {idx} on GPU {gpu_id}: {e}")
            results.append(None)
    
    return results

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
@click.option('--workers',    help='Number of parallel workers for image processing', metavar='INT', type=int, default=32, show_default=True)
@click.option('--batch-size', help='Number of images to process in each batch', metavar='INT', type=int, default=1000, show_default=True)

def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    workers: int,
    batch_size: int
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Parallelization and Memory Management:

    Use --workers to control the number of CPU cores used for image processing
    (default: 32). Use --batch-size to control how many images are processed
    in each batch (default: 1000). Larger batch sizes use more memory but may
    be more efficient. For very large datasets (like ImageNet), consider using
    smaller batch sizes to avoid memory issues.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal \\
        --workers=32
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    # Validate transform parameters
    if transform in ['center-crop', 'center-crop-wide', 'center-crop-dhariwal']:
        if resolution is None:
            raise click.ClickException(f'must specify --resolution=WxH when using {transform} transform')
        if transform == 'center-crop-dhariwal' and resolution[0] != resolution[1]:
            raise click.ClickException('width and height must match in --resolution=WxH when using center-crop-dhariwal transform')
    
    dataset_attrs = None

    # Process images in batches to avoid loading everything into memory
    output_width, output_height = resolution if resolution is not None else (None, None)
    labels = []
    
    print(f"Processing {num_files} images in batches of {batch_size} with {workers} workers...")
    
    with mp.Pool(workers) as pool:
        batch = []
        batch_start_idx = 0
        
        for idx, image in tqdm(enumerate(input_iter), total=num_files, desc="Processing images"):
            batch.append((idx, image, transform, output_width, output_height))
            
            # Process batch when it's full or we've reached the end
            if len(batch) == batch_size or idx == num_files - 1:
                # Process current batch in parallel
                batch_results = pool.map(transform_image_worker, batch)
                
                # Filter valid results and process them
                valid_batch_results = [result for result in batch_results if result is not None]
                valid_batch_results.sort(key=lambda x: x[0])  # Sort by index
                
                # Save results from this batch
                for result_idx, img, label in valid_batch_results:
                    idx_str = f'{result_idx:08d}'
                    archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

                    # Error check to require uniform image attributes across
                    # the whole dataset.
                    assert img.ndim == 3
                    cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0]}
                    if dataset_attrs is None:
                        dataset_attrs = cur_image_attrs
                        width = dataset_attrs['width']
                        height = dataset_attrs['height']
                        if width != height:
                            raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                        if width != 2 ** int(np.floor(np.log2(width))):
                            raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
                    elif dataset_attrs != cur_image_attrs:
                        err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
                        raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

                    # Save the image as an uncompressed PNG.
                    img = PIL.Image.fromarray(img)
                    image_bits = io.BytesIO()
                    img.save(image_bits, format='png', compress_level=0, optimize=False)
                    save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
                    labels.append([archive_fname, label] if label is not None else None)
                
                # Clear batch for next iteration
                batch = []

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--gpus',       help='Number of GPUs to use for parallel encoding', metavar='INT', type=int, default=8, show_default=True)
@click.option('--batch-size', help='Number of images per GPU in each batch', metavar='INT', type=int, default=100, show_default=True)
@click.option('--pack-shards', is_flag=True, default=False, help='After encoding, pack individual .npy files into contiguous shards (creates vae_packed).')
@click.option('--shard-size-gb', type=float, default=4.0, show_default=True, help='Target shard size in GB when --pack-shards is used.')

def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    gpus: int,
    batch_size: int,
    pack_shards: bool,
    shard_size_gb: float,
):
    """Encode pixel data to VAE latents.
    
    Parallelization and Memory Management:
    
    Use --gpus to control the number of GPUs used for parallel encoding
    (default: 8). Use --batch-size to control how many images each GPU
    processes in each batch (default: 100). Images are distributed across
    GPUs in round-robin fashion and processed in batches to avoid loading
    all images into memory at once.
    
    Example:
    \b
    python dataset_tool.py encode --source=datasets/img64.zip \\
        --dest=datasets/img64_encoded.zip --gpus=8 --batch-size=50
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    # Process images in batches across GPUs to avoid loading everything into memory
    labels = []
    
    print(f"Processing {num_files} images in batches of {batch_size * gpus} across {gpus} GPUs...")
    
    with mp.Pool(gpus) as pool:
        batch = []
        gpu_batches = [[] for _ in range(gpus)]
        
        for idx, image in tqdm(enumerate(input_iter), total=num_files, desc="Encoding images"):
            # Distribute images across GPUs in round-robin fashion
            gpu_id = idx % gpus
            gpu_batches[gpu_id].append((idx, image))
            
            # Process when any GPU batch is full or we've reached the end
            max_batch_size = max(len(gpu_batch) for gpu_batch in gpu_batches)
            if max_batch_size >= batch_size or idx == num_files - 1:
                # Prepare arguments for each GPU
                gpu_args = []
                for gpu_id, gpu_batch in enumerate(gpu_batches):
                    if gpu_batch:  # Only process non-empty batches
                        gpu_args.append((gpu_id, gpu_batch, model_url))
                
                # Process current batches in parallel across GPUs
                if gpu_args:
                    batch_results = pool.map(encode_image_worker, gpu_args)
                    
                    # Flatten results and sort by index
                    current_results = []
                    for gpu_result in batch_results:
                        current_results.extend([r for r in gpu_result if r is not None])
                    current_results.sort(key=lambda x: x[0])
                    
                    # Save results from this batch
                    for result_idx, mean_std, label in current_results:
                        idx_str = f'{result_idx:08d}'
                        archive_fname = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

                        f = io.BytesIO()
                        np.save(f, mean_std)
                        save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
                        labels.append([archive_fname, label] if label is not None else None)
                
                # Clear batches for next iteration
                gpu_batches = [[] for _ in range(gpus)]

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

    # Optional shard packing (only if dest is a folder; if zip, user should unzip first or re-run packing manually)
    if pack_shards:
        if file_ext(dest) == 'zip':
            print('[encode] --pack-shards ignored because destination is a .zip. Unzip and run pack manually if needed.')
        else:
            try:
                from preprocessing.dataset_tools import pack_npy_directory as _pack
            except Exception:
                from dataset_tools import pack_npy_directory as _pack  # fallback if run from within package
            packed_dir = os.path.join(dest, 'vae_packed')
            print(f'[encode] Packing encoded VAE latents into shards at {packed_dir} (target {shard_size_gb} GB)...')
            _pack(dest, packed_dir, pattern='mean-std-', shard_size_gb=shard_size_gb)
            print('[encode] Packing complete.')

#----------------------------------------------------------------------------
# ===== ADD BELOW TO YOUR EXISTING DATASET PREPARATION SCRIPT =====
from typing import List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from transformers import AutoImageProcessor, AutoModel

@dataclass
class ImageRef:
    kind: str            # 'folder' or 'zip'
    path: str            # abs path for folder; inner path for zip
    zip_path: Optional[str] = None
    label: Optional[int] = None

def open_image_folder_refs(source_dir: str, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageRef]]:
    input_images = []
    def _recurse(root: str):
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse(os.path.join(root, e.name))
    _recurse(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])
    arch_fnames = {f: os.path.relpath(f, source_dir).replace('\\','/') for f in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    labels = {}
    meta = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta):
        with open(meta, 'r') as fh:
            data = json.load(fh).get('labels')
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    if not labels:
        tops = {a: a.split('/')[0] if '/' in a else '' for a in arch_fnames.values()}
        idxs = {n:i for i,n in enumerate(sorted(set(tops.values())))}
        if len(idxs) > 1:
            labels = {a: idxs[tops[a]] for a in tops}

    def it():
        for f in input_images[:max_idx]:
            yield ImageRef('folder', os.path.abspath(f), None, labels.get(arch_fnames[f]))
    return max_idx, it()

def open_image_zip_refs(source: str, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageRef]]:
    with zipfile.ZipFile(source, 'r') as z:
        imgs = [s for s in sorted(z.namelist()) if is_image_ext(s)]
        max_idx = maybe_min(len(imgs), max_images)
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json','r') as fh:
                data = json.load(fh).get('labels')
                if data is not None:
                    labels = {x[0]: x[1] for x in data}
    def it():
        for s in imgs[:max_idx]:
            yield ImageRef('zip', s, os.path.abspath(source), labels.get(s))
    return max_idx, it()

def open_dataset_refs(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder_refs(source, max_images=max_images)
    if os.path.isfile(source) and file_ext(source) == 'zip':
        return open_image_zip_refs(source, max_images=max_images)
    raise click.ClickException(f'Unsupported or missing source: {source}')

def _atomic_save_npy(final_path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp = final_path + ".tmp"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass
    with open(tmp, "wb") as f:
        np.save(f, arr)  # using a file handle prevents ".npy" auto-append
    os.replace(tmp, final_path)

def _dinov3_worker_loop(
    gpu_id: int,
    model_name: str,
    dtype_str: str,
    decode_threads: int,
    serialize_threads: int,
    write_root: str,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
):
    # reduce CPU oversubscription
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    try:
        import PIL.ImageFile as _P
        _P.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        pass

    from concurrent.futures import ThreadPoolExecutor

    zf_cache: dict[str, zipfile.ZipFile] = {}

    def _open_from_ref(ref: ImageRef) -> PIL.Image.Image:
        if ref.kind == 'folder':
            return PIL.Image.open(ref.path).convert('RGB')
        if ref.zip_path not in zf_cache:
            zf_cache[ref.zip_path] = zipfile.ZipFile(ref.zip_path, 'r')
        with zf_cache[ref.zip_path].open(ref.path, 'r') as f:
            return PIL.Image.open(f).convert('RGB')


    def _write_item(idx: int, arr_hidden_states: np.ndarray, label: Optional[int]):
        idx_str = f'{idx:08d}'
        rel_hidden = f'{idx_str[:5]}/img-dinov3-{idx_str}_hidden.npy'
        full_hidden = os.path.join(write_root, rel_hidden)
        _atomic_save_npy(full_hidden, arr_hidden_states)
        # Return the hidden states path in labels
        return (idx, rel_hidden, label)

    try:
        torch.cuda.set_device(gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("medium")
        except Exception: pass

        device = torch.device(f'cuda:{gpu_id}')
        processor = AutoImageProcessor.from_pretrained(model_name)
        # load in target dtype to avoid conversions
        torch_dtype = torch.float16 if dtype_str in ('float16','bfloat16') else torch.float32
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        if dtype_str == 'bfloat16' and model.dtype != torch.bfloat16:
            model = model.bfloat16()
        model.eval()

        with torch.inference_mode():
            _ = model(pixel_values=torch.zeros(1,3,16,16, device=device, dtype=model.dtype))
        torch.backends.cudnn.benchmark = True

        out_dtype = np.float16 if dtype_str in ('float16','bfloat16') else np.float32

        decode_pool = ThreadPoolExecutor(max_workers=max(1, decode_threads))
        serial_pool = ThreadPoolExecutor(max_workers=max(1, serialize_threads))
    except Exception as e:
        out_queue.put(('__init_error__', gpu_id, str(e)))
        return

    pending_serial = []

    def _flush_serial(max_take: Optional[int] = None):
        ready = []
        still = []
        for fut in pending_serial:
            if fut.done():
                try: ready.append(fut.result())
                except Exception as ex: print(f"[W{gpu_id}] serialize error: {ex}")
            else:
                still.append(fut)
        pending_serial[:] = still
        if ready:
            CHUNK = 512
            for i in range(0, len(ready), CHUNK):
                out_queue.put(('batch', gpu_id, ready[i:i+CHUNK]))

    try:
        DO_RESIZE = True
        DO_CENTER_CROP = True

        while True:
            batch = in_queue.get()
            if batch is None:
                break

            # Stage A: parallel decode
            futs = [decode_pool.submit(_open_from_ref, ref) for (_, ref) in batch]
            pil_images, indices, labels_local = [], [], []
            for (idx, ref), fut in zip(batch, futs):
                img = fut.result()
                pil_images.append(img); indices.append(idx); labels_local.append(ref.label)

            # Stage B/C: processor + GPU
            with torch.inference_mode():
                inputs = processor(
                    images=pil_images,
                    return_tensors='pt',
                    do_resize=DO_RESIZE,
                    do_center_crop=DO_CENTER_CROP,
                )
                pv = inputs['pixel_values'].contiguous(memory_format=torch.channels_last).pin_memory()
                pv = pv.to(device, non_blocking=True, dtype=model.dtype)
                outputs = model(pixel_values=pv, output_hidden_states=False)
                hidden = outputs.last_hidden_state  # [B, 1+R+N, D]
                hidden_cpu = hidden.detach().to('cpu')
                if out_dtype == np.float16 and hidden_cpu.dtype != torch.float16:
                    hidden_cpu = hidden_cpu.to(torch.float16)

            # Stage D: write to disk in thread pool
            B = hidden_cpu.shape[0]
            for b in range(B):
                h = hidden_cpu[b]             # [1+R+N, D]
                # Store the entire hidden states array
                if h.dtype != np.float16 and out_dtype == np.float16:
                    h = h.contiguous().numpy().astype(np.float16, copy=False)
                else:
                    h = h.contiguous().numpy()
                fut = serial_pool.submit(_write_item, indices[b], h, labels_local[b])
                pending_serial.append(fut)

            _flush_serial()

        # finalize
        while pending_serial:
            _flush_serial()

        decode_pool.shutdown(wait=True, cancel_futures=False)
        serial_pool.shutdown(wait=True, cancel_futures=False)
        for z in zf_cache.values():
            try: z.close()
            except Exception: pass

    except Exception as e:
        out_queue.put(('error', gpu_id, str(e), []))

@cmdline.command(name='encode-dinov3')
@click.option('--model-name',        type=str, default='facebook/dinov3-vit7b16-pretrain-lvd1689m', show_default=True)
@click.option('--source',            type=str, required=True)
@click.option('--dest',              type=str, required=True)
@click.option('--max-images',        type=int)
@click.option('--gpus',              type=int, default=8, show_default=True)
@click.option('--batch-size',        type=int, default=100, show_default=True)
@click.option('--dtype',             type=click.Choice(['float32','float16','bfloat16']), default='float16', show_default=True)
@click.option('--decode-threads',    type=int, default=4, show_default=True)
@click.option('--serialize-threads', type=int, default=2, show_default=True)
@click.option('--pack-shards', is_flag=True, default=False, help='After encoding, pack individual *_hidden.npy files into contiguous shards (creates dinov3_packed).')
@click.option('--shard-size-gb', type=float, default=4.0, show_default=True, help='Target shard size in GB when --pack-shards is used.')
def encode_dinov3(
    model_name: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    gpus: int,
    batch_size: int,
    dtype: str,
    decode_threads: int,
    serialize_threads: int,
    pack_shards: bool,
    shard_size_gb: float,
):
    """
    DINOv3 precompute storing full hidden states, forced no-resize, and worker-local I/O.
    Workers write artifacts directly to disk; main process only assembles labels and (optionally) zips once.
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest must not be empty')

    num_files, ref_iter = open_dataset_refs(source, max_images=max_images)

    # If dest is .zip, write to a temp folder first; zip at the end.
    import tempfile, shutil
    is_zip_dest = (file_ext(dest).lower() == 'zip')
    if is_zip_dest:
        write_root = tempfile.mkdtemp(prefix="dinov3_tmp_")
        archive_root_dir = write_root
        def finalize_zip():
            with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_STORED) as z:
                # Write all files (including dataset.json)
                for root, _, files in os.walk(write_root):
                    for f in files:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, write_root).replace('\\','/')
                        z.write(full, arcname=rel)
            shutil.rmtree(write_root, ignore_errors=True)
    else:
        write_root = dest
        os.makedirs(write_root, exist_ok=True)
        archive_root_dir = write_root
        def finalize_zip(): pass

    print(f"[encode-dinov3] {num_files} images → {dest} | gpus={gpus} bs={batch_size} dtype={dtype}")

    in_queues = [mp.Queue(maxsize=16) for _ in range(gpus)]
    out_queue = mp.Queue()

    procs = []
    for gid in range(gpus):
        p = mp.Process(
            target=_dinov3_worker_loop,
            args=(gid, model_name, dtype, decode_threads, serialize_threads, write_root, in_queues[gid], out_queue)
        )
        p.daemon = True
        p.start()
        procs.append(p)

    # dispatch
    staging: List[List[Tuple[int, ImageRef]]] = [[] for _ in range(gpus)]
    images_dispatched = 0
    for idx, ref in tqdm(enumerate(ref_iter), total=num_files, desc="DINOv3 dispatch"):
        gid = idx % gpus
        staging[gid].append((idx, ref))
        images_dispatched += 1
        if len(staging[gid]) >= batch_size:
            in_queues[gid].put(staging[gid]); staging[gid] = []

    for gid in range(gpus):
        if staging[gid]:
            in_queues[gid].put(staging[gid])
    for q in in_queues:
        q.put(None)

    # collect minimal metadata (no payloads)
    import queue as _q
    labels_dict = {}
    images_done = 0
    while images_done < images_dispatched:
        try:
            msg = out_queue.get(timeout=1.0)
        except _q.Empty:
            continue
        kind = msg[0]
        if kind == '__init_error__':
            _, gpu_failed, err = msg
            raise RuntimeError(f"Worker {gpu_failed} failed: {err}")
        if kind == 'batch':
            _, gpu_id_r, result_list = msg
            for idx_i, rel_path, label in result_list:
                labels_dict[idx_i] = [rel_path, label] if label is not None else None
                images_done += 1
        elif kind == 'error':
            _, gpu_id_r, err, _ = msg
            print(f"[encode-dinov3] Worker {gpu_id_r} error: {err}")
        else:
            print(f"[encode-dinov3] Unknown message {kind}")

    for p in procs:
        p.join()

    # build labels list in order and write dataset.json to write_root
    labels = [labels_dict[i] for i in range(images_dispatched)]
    with open(os.path.join(archive_root_dir, 'dataset.json'), 'w') as fh:
        json.dump({'labels': labels if all(x is not None for x in labels) else None}, fh)

    finalize_zip()
    print(f"[encode-dinov3] Done. Encoded {images_done}/{images_dispatched} images.")

    # Optional shard packing (only if dest is a folder; if zip, user should unzip first or re-run packing manually)
    if pack_shards:
        if file_ext(dest) == 'zip':
            print('[encode-dinov3] --pack-shards ignored because destination is a .zip. Unzip and run pack manually if needed.')
        else:
            try:
                from preprocessing.dataset_tools import pack_npy_directory as _pack
            except Exception:
                from dataset_tools import pack_npy_directory as _pack
            packed_dir = os.path.join(dest, 'dinov3_packed')
            print(f'[encode-dinov3] Packing DINOv3 hidden states into shards at {packed_dir} (target {shard_size_gb} GB)...')
            _pack(dest, packed_dir, pattern='_hidden.npy', shard_size_gb=shard_size_gb)
            print('[encode-dinov3] Packing complete.')
# ===== END ADD =====




# ================= Additional optimization: pack existing per-image npy into shards =================
# Lightweight utility (CLI separate from click group to avoid large refactor in truncated file version).
def pack_npy_directory(src_dir: str, dest_dir: str, pattern: str = '_hidden.npy', shard_size_gb: float = 4.0):
    """Pack many fixed-shape .npy arrays into large binary shards with an index.json.

    src_dir: directory containing .npy files (e.g. DINO embeddings or VAE latents)
    dest_dir: output directory (created if missing)
    pattern: substring filter for file selection
    shard_size_gb: target maximum shard size
    """
    import numpy as _np, json as _json, os as _os, math as _math, glob as _glob
    _os.makedirs(dest_dir, exist_ok=True)
    files = sorted([f for f in _glob.glob(_os.path.join(src_dir, '**', '*.npy'), recursive=True) if pattern in _os.path.basename(f)])
    assert files, f'No files found matching pattern {pattern}'
    # Probe first file for shape/dtype
    sample = _np.load(files[0], mmap_mode='r')
    shape = sample.shape
    dtype = sample.dtype
    rec_size = sample.nbytes
    max_shard_bytes = int(shard_size_gb * (1024**3))
    per_shard = max(1, max_shard_bytes // rec_size)
    shards = []
    cur_data = bytearray()
    shard_index = 0
    shard_count = 0
    shard_meta = []
    labels = []  # attempt to load sibling dataset.json if present
    # Try labels
    label_file = _os.path.join(src_dir, 'dataset.json')
    label_map = None
    if _os.path.isfile(label_file):
        try:
            with open(label_file, 'r') as fh:
                ld = _json.load(fh)['labels']
            if ld is not None:
                label_map = dict(ld)
        except Exception:
            pass
    def flush():
        nonlocal cur_data, shard_index, shard_count
        if not cur_data:
            return
        fname = f'shard_{shard_index:05d}.bin'
        with open(_os.path.join(dest_dir, fname), 'wb') as fw:
            fw.write(cur_data)
        shard_meta.append({'file': fname, 'samples': shard_count})
        shard_index += 1
        shard_count = 0
        cur_data = bytearray()
    for i, f in enumerate(files):
        arr = _np.load(f, mmap_mode='r')
        assert arr.shape == shape and arr.dtype == dtype, 'Inconsistent array shape/dtype'
        cur_data.extend(arr.tobytes(order='C'))
        shard_count += 1
        # gather label if possible
        if label_map is not None:
            # Normalize path relative to src_dir like existing dataset.json style
            rel = _os.path.relpath(f, src_dir).replace('\\','/')
            labels.append(label_map.get(rel, -1))
        else:
            labels.append(-1)
        if shard_count >= per_shard or len(cur_data) >= max_shard_bytes:
            flush()
    flush()
    index = {
        'dtype': str(dtype),
        'shape': list(shape),
        'record_size': rec_size,
        'shards': shard_meta,
        'labels': labels,
    }
    with open(_os.path.join(dest_dir, 'index.json'), 'w') as fh:
        _json.dump(index, fh)
    print(f"Packed {len(files)} records into {len(shard_meta)} shards at {dest_dir}")


# ================= Parallel & Resumable Shard Packer =================
def pack_npy_directory_parallel(
    src_dir: str,
    dest_dir: str,
    pattern: str = '_hidden.npy',
    shard_size_gb: float = 4.0,
    workers: int = 8,
    resume: bool = False,
):
    """Pack many fixed-shape .npy arrays into large binary shards (parallel + resumable).

    Strategy:
      * Enumerate & sort matching files.
      * Determine record_size from first file (shape/dtype must match for all).
      * Compute target samples per shard from shard_size_gb.
      * Split file list into contiguous shard segments. Each worker processes shard segments assigned to it.
      * Each shard is written sequentially to avoid random writes; within shard we preserve order.
      * Resume: if dest_dir contains existing shard_XXXXX.bin files, we infer their sample counts from file sizes and skip those files.

    Produces dest_dir/index.json with same schema as original packer.
    """
    import numpy as _np, json as _json, os as _os, glob as _glob, math as _math, multiprocessing as _mp
    from functools import partial as _partial

    _os.makedirs(dest_dir, exist_ok=True)
    all_files = sorted([f for f in _glob.glob(_os.path.join(src_dir, '**', '*.npy'), recursive=True) if pattern in _os.path.basename(f)])
    if not all_files:
        raise FileNotFoundError(f'No npy files matching pattern {pattern} in {src_dir}')

    # Probe first non-empty file
    probe = _np.load(all_files[0], mmap_mode='r')
    shape = probe.shape; dtype = probe.dtype; rec_size = probe.nbytes
    max_shard_bytes = int(shard_size_gb * (1024**3))
    per_shard = max(1, max_shard_bytes // rec_size)

    # Resume detection
    existing_shards = sorted([f for f in os.listdir(dest_dir) if f.startswith('shard_') and f.endswith('.bin')]) if resume else []
    shard_meta = []
    labels = []
    processed_files = 0
    if existing_shards:
        print(f'[pack-par] Resume enabled: found {len(existing_shards)} existing shard files, validating...')
        for shf in existing_shards:
            full = os.path.join(dest_dir, shf)
            size = os.path.getsize(full)
            if size % rec_size != 0:
                raise RuntimeError(f'Existing shard {shf} size {size} not divisible by record_size {rec_size}')
            samples = size // rec_size
            shard_idx = int(shf.split('_')[1].split('.')[0])
            shard_meta.append({'file': shf, 'samples': samples})
            processed_files += samples
        shard_meta.sort(key=lambda x: x['file'])
        print(f'[pack-par] Will skip first {processed_files} files already packed.')

    remaining_files = all_files[processed_files:]
    if not remaining_files:
        print('[pack-par] Nothing to do; all files already packed.')
        # Attempt to load labels from existing index; otherwise fill -1
        index_path = os.path.join(dest_dir, 'index.json')
        if os.path.isfile(index_path):
            return
    total_files = len(all_files)

    # Labels
    label_file = os.path.join(src_dir, 'dataset.json')
    label_map = None
    if os.path.isfile(label_file):
        try:
            with open(label_file,'r') as fh:
                ld = _json.load(fh)['labels']
            if ld is not None:
                label_map = dict(ld)
        except Exception:
            pass

    # Pre-size labels list (lazy fill with -1 then fill for new part only) to avoid large concatenations
    # We only assemble labels for new part then if existing index.json exists we merge; else we create full list.
    existing_labels = None
    existing_index_path = os.path.join(dest_dir, 'index.json')
    if resume and os.path.isfile(existing_index_path):
        try:
            with open(existing_index_path,'r') as fh:
                existing_index = _json.load(fh)
            existing_labels = existing_index.get('labels')
        except Exception:
            existing_labels = None

    def _collect_labels(files_slice):
        out = []
        if label_map is None:
            return [-1]*len(files_slice)
        for f in files_slice:
            rel = os.path.relpath(f, src_dir).replace('\\','/')
            out.append(label_map.get(rel, -1))
        return out

    # Build shard plan for remaining files
    rem_total = len(remaining_files)
    if rem_total == 0:
        pass
    else:
        num_new_shards = _math.ceil(rem_total / per_shard)
        next_shard_id = len(shard_meta)
        shard_jobs = []  # list of job dicts
        for i in range(num_new_shards):
            start = i * per_shard
            end = min(rem_total, (i + 1) * per_shard)
            shard_jobs.append({
                'shard_id': next_shard_id + i,
                'files': remaining_files[start:end],
            })

        # Prepare static payload for workers
        worker_args = []
        for job in shard_jobs:
            worker_args.append((job['shard_id'], job['files'], dest_dir, shape, str(dtype), rec_size))

        print(f'[pack-par] Packing {rem_total} remaining records using {workers} workers into ~{num_new_shards} shards (target {per_shard} rec/shard)...')
        # Import top-level worker (defined below) dynamically to avoid circular reference issues
        _worker = parallel_pack_worker  # self module import
        with _mp.Pool(processes=workers) as pool:
            for shard_id, out_name, samples in sorted(pool.imap_unordered(_worker, worker_args), key=lambda x: x[0]):
                shard_meta.append({'file': out_name, 'samples': samples})
                # collect labels for this shard
                labels_part = _collect_labels([f for f in remaining_files if f.endswith(out_name)][:0])  # placeholder (labels gathered separately below)
        # Because we collected no labels per-file in worker (to minimize IPC), gather labels directly now for remaining files in order
        if label_map is not None:
            for f in remaining_files:
                rel = os.path.relpath(f, src_dir).replace('\\','/')
                labels.append(label_map.get(rel, -1))
        else:
            labels.extend([-1]*rem_total)

    # Combine labels (existing + new)
    if existing_labels is not None:
        # existing_labels length should equal processed_files; if not warn and rebuild
        if len(existing_labels) != processed_files:
            print(f'[pack-par] Warning: existing labels length {len(existing_labels)} != processed files {processed_files}; rebuilding label list with -1 for old part.')
            full_labels = [-1]*processed_files + labels
        else:
            full_labels = existing_labels + labels
    else:
        # Need labels for entire dataset. If we processed from scratch processed_files==0
        if processed_files > 0:
            # Fill old part with -1 if we don't have them
            full_labels = [-1]*processed_files + labels
        else:
            full_labels = labels

    index = {
        'dtype': str(dtype),
        'shape': list(shape),
        'record_size': rec_size,
        'shards': shard_meta,
        'labels': full_labels,
    }
    with open(os.path.join(dest_dir, 'index.json'), 'w') as fh:
        _json.dump(index, fh)
    print(f"[pack-par] Done. Packed {len(all_files)} records into {len(shard_meta)} shards at {dest_dir}")


# Top-level worker function for parallel shard packing (must be picklable)
def parallel_pack_worker(args):
    import numpy as _np, os as _os
    shard_id, files, dest_dir, shape, dtype_str, rec_size = args
    out_name = f'shard_{shard_id:05d}.bin'
    out_path = _os.path.join(dest_dir, out_name)
    if _os.path.exists(out_path):
        size = _os.path.getsize(out_path)
        samples = size // rec_size
        return (shard_id, out_name, samples)
    dtype = _np.dtype(dtype_str)
    with open(out_path, 'wb') as fw:
        for f in files:
            arr = _np.load(f, mmap_mode='r')
            if arr.shape != tuple(shape) or arr.dtype != dtype:
                raise ValueError(f'Shape/dtype mismatch in {f}: {arr.shape}/{arr.dtype} != {shape}/{dtype}')
            fw.write(arr.tobytes(order='C'))
    return (shard_id, out_name, len(files))


# Register a lightweight CLI command for packing to avoid heavy click group dependency
@cmdline.command(name='pack')
@click.option('--src', type=str, required=True, help='Source directory containing per-sample .npy files.')
@click.option('--dest', type=str, required=True, help='Destination directory for packed shards.')
@click.option('--pattern', type=str, default='_hidden.npy', show_default=True, help='Substring pattern to select .npy files.')
@click.option('--shard-size-gb', type=float, default=4.0, show_default=True, help='Target shard size in GB.')
@click.option('--workers', type=int, default=8, show_default=True, help='Parallel workers (processes).')
@click.option('--resume', is_flag=True, default=False, help='Resume from existing shards (append remaining).')
@click.option('--parallel/--no-parallel', default=True, help='Disable parallel mode (fallback to original single-thread pack).')
def pack_cli(src: str, dest: str, pattern: str, shard_size_gb: float, workers: int, resume: bool, parallel: bool):
    """Pack per-sample .npy arrays into shards (parallel & resumable)."""
    if parallel:
        pack_npy_directory_parallel(src, dest, pattern=pattern, shard_size_gb=shard_size_gb, workers=workers, resume=resume)
    else:
        if resume:
            print('[pack] Resume requested but not supported in non-parallel mode; ignoring.')
        pack_npy_directory(src, dest, pattern=pattern, shard_size_gb=shard_size_gb)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing
    cmdline()