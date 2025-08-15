# dataset.py

import os
import json
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    """
    Original dataset with optional DINOv3 tokens loading.

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
        dinov3_subdir: Optional[str] = "dinov3-vit7b16",  # folder or zip extracted folder name
    ):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy', '.npz'}

        self.images_dir = os.path.join(data_dir, 'imagenet_256_vae')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        self.load_dinov3 = load_dinov3
        self.dinov3_dir = None
        if load_dinov3:
            self.dinov3_dir = os.path.join(data_dir, dinov3_subdir)
            if not os.path.isdir(self.dinov3_dir):
                # allow pointing directly to the path (zip extracted or plain dir)
                self.dinov3_dir = dinov3_subdir

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
        }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
        )

        # vae features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
        }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
        )

        if self.load_dinov3:
            self._dinov3_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self.dinov3_dir)
                for root, _dirs, files in os.walk(self.dinov3_dir) for fname in files
            }
            # only .npz from our encoder
            self.dinov3_fnames = sorted(
                fname for fname in self._dinov3_fnames if self._file_ext(fname) in supported_ext and fname.endswith('.npz')
            )
        else:
            self.dinov3_fnames = None

        # labels come from the VAE features' dataset.json (unchanged)
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
                "DINOv3 count must match images count"
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "VAE/Images count mismatch"

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        # raw image -> CHW uint8
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                img = pyspng.load(f.read())  # HWC
                image = img.reshape(*img.shape[:2], -1).transpose(2, 0, 1)
            else:
                img = np.array(PIL.Image.open(f).convert('RGB'))  # HWC
                image = img.reshape(*img.shape[:2], -1).transpose(2, 0, 1)

        vae_latents = np.load(os.path.join(self.features_dir, feature_fname))

        label = torch.tensor(self.labels[idx])

        if not self.load_dinov3:
            return torch.from_numpy(image), torch.from_numpy(vae_latents), label

        dino_fname = self.dinov3_fnames[idx]
        with np.load(os.path.join(self.dinov3_dir, dino_fname)) as z:
            tokens = z['tokens']  # [1+N, D], tokens[0] is CLS
            cls = z['cls']        # [D]
        # torchify
        return (
            torch.from_numpy(image),
            torch.from_numpy(vae_latents),
            label,
            torch.from_numpy(tokens),
            torch.from_numpy(cls),
        )
