# dataset.py

import os
import json
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
import mmap

# Optimized dataset that can fall back to legacy per-file .npy but prefers
# memory-mapped packed shards produced by the new `pack-npy` tool.


class _PackedArray:
    """Helper for fixed-size records stored across shards using mmap.

    Expects an index JSON with fields: dtype, shape, record_size, shards(list of {file,samples}), labels(optional array).
    """
    def __init__(self, root_dir: str, index_file: str):
        with open(index_file, 'r') as f:
            meta = json.load(f)
        self.dtype = np.dtype(meta['dtype'])
        self.shape = tuple(meta['shape'])
        self.record_size = int(meta['record_size'])
        self.shards_meta = meta['shards']
        self.cum_counts: List[Tuple[int,int]] = []  # (start,count)
        cur = 0
        for sh in self.shards_meta:
            self.cum_counts.append((cur, sh['samples']))
            cur += sh['samples']
        self.count = cur
        self.root_dir = root_dir
        self._mmaps: List[mmap.mmap] = [None]*len(self.shards_meta)
        self._files: List[object] = [None]*len(self.shards_meta)  # keep file handles
        self.labels = None
        if meta.get('labels') is not None:
            self.labels = np.array(meta['labels'])

    def __len__(self):
        return self.count

    def _locate(self, idx: int) -> Tuple[int,int,int]:
        # binary search since shard count is small (but implement linear fallback)
        lo, hi = 0, len(self.cum_counts)-1
        while lo <= hi:
            mid = (lo+hi)//2
            start, cnt = self.cum_counts[mid]
            if idx < start:
                hi = mid-1
            elif idx >= start+cnt:
                lo = mid+1
            else:
                return mid, start, cnt
        raise IndexError

    def get(self, idx: int) -> np.ndarray:
        shard_idx, shard_start, _ = self._locate(idx)
        rel = idx - shard_start
        shard_info = self.shards_meta[shard_idx]
        if self._mmaps[shard_idx] is None:
            path = os.path.join(self.root_dir, shard_info['file'])
            f = open(path, 'rb')
            self._files[shard_idx] = f
            self._mmaps[shard_idx] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mmaps[shard_idx]
        off = rel * self.record_size
        view = memoryview(mm)[off: off + self.record_size]
        arr = np.frombuffer(view, dtype=self.dtype)
        return arr.reshape(self.shape)

    def get_label(self, idx: int):
        if self.labels is None:
            return -1
        return self.labels[idx]


class PackedFeaturesDataset(Dataset):
    """Dataset that ONLY supports packed shard format (no legacy per-file .npy fallback).

    Required directory structure:
        data_dir/
          vae_packed/index.json
          vae_packed/shard_00000.bin
          dinov3_packed/index.json
          dinov3_packed/shard_00000.bin

    Each index.json contains:
        {
          "dtype": "float16", "shape": [C,H,W] or [T,D], "record_size": int,
          "shards": [ {"file": "shard_00000.bin", "samples": N}, ...],
          "labels": [...]
        }

    __getitem__ returns (vae_latents, label, dino_patches, dino_cls_tokens)
    where dino_cls_tokens is the CLS token split into cls_tokens pieces.
    """
    def __init__(self,
                 data_dir: str,
                 cls_tokens: int = 16,
                 dinov3_register_tokens: int = 4):
        self.data_dir = data_dir
        self.cls_tokens = cls_tokens
        self.reg_tokens = dinov3_register_tokens
        self._use_packed = True  # only mode now

        vae_packed_dir = os.path.join(data_dir, 'vae_packed')
        dino_packed_dir = os.path.join(data_dir, 'dinov3_packed')
        vae_index = os.path.join(vae_packed_dir, 'index.json')
        dino_index = os.path.join(dino_packed_dir, 'index.json')

        if not (os.path.isfile(vae_index) and os.path.isfile(dino_index)):
            raise FileNotFoundError(
                'Packed shards not found. Expected both VAE and DINO shard indexes. '\
                'Please run preprocessing to create vae_packed/ and dinov3_packed/ under the data directory.'
            )

        self.vae = _PackedArray(vae_packed_dir, vae_index)
        self.dino = _PackedArray(dino_packed_dir, dino_index)
        assert len(self.vae) == len(self.dino), 'Packed VAE/DINO count mismatch'
        self.length = len(self.vae)
        if self.vae.labels is not None:
            self.labels = self.vae.labels.astype(np.int64)
        elif self.dino.labels is not None:
            self.labels = self.dino.labels.astype(np.int64)
        else:
            self.labels = np.zeros(self.length, dtype=np.int64) - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vae_arr = self.vae.get(idx)
        dino_arr = self.dino.get(idx)  # shape [1+R+N, D]
        label = int(self.labels[idx])

        # Separate CLS and patch tokens (skip register tokens)
        cls_token = dino_arr[0]
        patches = dino_arr[1 + self.reg_tokens:]
        cls_dim = cls_token.shape[0]
        assert cls_dim % self.cls_tokens == 0, 'CLS dim not divisible by cls_tokens'
        cls_split = cls_token.reshape(self.cls_tokens, cls_dim // self.cls_tokens)

        return (
            torch.from_numpy(vae_arr).contiguous(),
            torch.tensor(label, dtype=torch.int64),
            torch.from_numpy(patches).contiguous(),
            torch.from_numpy(cls_split).contiguous(),
        )


# For backward compatibility keep original class name pointing to optimized implementation
CustomDataset = PackedFeaturesDataset
