# dataset.py

import os
import json
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np

import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    """
    Dataset with optional DINOv3 tokens loading (NPY-only).

    Returns:
      if load_dinov3 == False:
        (raw_image[3,H,W] uint8, vae_latents[np], label[int64])

      if load_dinov3 == True:
        (raw_image[3,H,W] uint8, vae_latents[np], label[int64],
         dinov3_tokens[1+N,D] float16/float32, dinov3_cls[D])
    """
    def __init__(
        self,
        data_dir: str,
        load_dinov3: bool = True,
        dinov3_subdir: Optional[str] = "dinov3-vit7b16",  # folder name or absolute path
        cls_tokens = 16,
    ):
        PIL.Image.init()
        supported_img_ext = set(PIL.Image.EXTENSION.keys()) | {'.npy'}  # allow .png/.jpg/... or .npy stubs

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        self.cls_tokens = cls_tokens
        self.load_dinov3 = load_dinov3
        self.dinov3_dir = None
        if load_dinov3:
            self.dinov3_dir = os.path.join(data_dir, dinov3_subdir)
            if not os.path.isdir(self.dinov3_dir):
                # allow passing the absolute path directly
                self.dinov3_dir = dinov3_subdir

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
        }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_img_ext
        )

        # VAE latents (.npy)
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
        }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) == '.npy'
        )

        # DINOv3 tokens (.npy only): we index by *_cls.npy; patches path is derived
        if self.load_dinov3:
            self._dinov3_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self.dinov3_dir)
                for root, _dirs, files in os.walk(self.dinov3_dir) for fname in files
            }
            self.dinov3_fnames = sorted(
                fname for fname in self._dinov3_fnames if fname.endswith('_hidden.npy')
            )
        else:
            self.dinov3_fnames = None

        # Labels come from the VAE features' dataset.json
        meta = os.path.join(self.features_dir, 'dataset.json')
        if not os.path.exists(meta):
            raise FileNotFoundError(f"Missing labels file: {meta}")
        with open(meta, 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        if self.load_dinov3:
            assert len(self.image_fnames) == len(self.dinov3_fnames), \
                f"DINOv3 count must match images count, got {len(self.dinov3_fnames)}, expected {len(self.image_fnames)}"
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "VAE/Images count mismatch"

    def _file_ext(self, fname: str) -> str:
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        feature_fname = self.feature_fnames[idx]

        # VAE latents (mean+std packed) — NPY mmap
        vae_latents = np.load(os.path.join(self.features_dir, feature_fname), mmap_mode='r')
        vae_latents = torch.from_numpy(vae_latents).to(torch.float32)

        label = torch.tensor(self.labels[idx])

        # DINOv3 cls + patches (both NPY). dataset.json (features) aligns by index.
        dino_hidden_rel = self.dinov3_fnames[idx]  # "..._cls.npy"
        dino_hidden_path = os.path.join(self.dinov3_dir, dino_hidden_rel)


        dino_hidden = np.load(dino_hidden_path)
        dino_hidden = torch.from_numpy(dino_hidden).to(torch.float32)
        dino_patches = dino_hidden[1+4:, :]
        dino_cls = dino_hidden[0:1, :]

        dino_cls = dino_cls.view(self.cls_tokens, -1)

        # torchify
        return (
            (vae_latents),
            label,
            (dino_patches),
            (dino_cls),
        )