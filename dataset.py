# dataset.py
# Sharded, memmapped dataset + DDP-aware shard-window sampler + test harness.
# Returns per sample:
#   if load_dinov3 == False:
#       (raw_or_placeholder, vae_latents, label)
#   if load_dinov3 == True and load_registers == False:
#       (raw_or_placeholder, vae_latents, label, dinov3_patches, dinov3_cls)
#   if load_dinov3 == True and load_registers == True:
#       (raw_or_placeholder, vae_latents, label, dinov3_patches, dinov3_cls, dinov3_registers)

import os, json, time, argparse
from typing import Optional, Tuple, Dict, Any, List, Iterator
from bisect import bisect_right
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm.auto import tqdm
from bisect import bisect_right
# --------------------------
# Global perf knobs (keep CPU threads modest in workers)
# --------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # helps on fixed shapes
except Exception:
    pass

def _worker_init_fn(_):
    # Ensure worker processes don't oversubscribe CPU threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass

# --------------------------
# Sharded memmap primitives
# --------------------------

class _ShardSpec:
    def __init__(self, root: str, meta_key: Optional[str], meta: Dict[str,Any]):
        self.root = root
        if meta_key is None:
            self.shape = tuple(meta["shape"])
            self.dtype = np.dtype(meta["dtype"])
            self.rps   = int(meta["records_per_shard"])
            self.shards = [(os.path.join(root, s["file"]), int(s["count"])) for s in meta["shards"]]
        else:
            m = meta[meta_key]
            self.shape = tuple(m["shape"])
            self.dtype = np.dtype(m["dtype"])
            self.rps   = int(m["records_per_shard"])
            self.shards = [(os.path.join(root, s["file"]), int(s["count"])) for s in m["shards"]]

class _ShardedArray:
    """
    Read-only memmap sharded array with small LRU of open shards.
    Uses robust index→(shard,offset) via offsets + binary search (no RPS assumptions).
    """
    def __init__(self, spec: _ShardSpec, cache: int = 16):
        self.spec = spec
        self.cache = max(1, int(cache))
        self._paths = [p for p,_ in spec.shards]
        self._counts = [c for _,c in spec.shards]

        # build offsets first
        self._offsets = []
        acc = 0
        for c in self._counts:
            self._offsets.append(acc)
            acc += c
        self.num = acc

        # now ends (exclusive)
        self._ends = [o + c for o, c in zip(self._offsets, self._counts)]

        self.rps = spec.rps
        self.sample_shape = spec.shape
        self.dtype = spec.dtype
        self._open = {}
        self._lru: List[int] = []

    def __len__(self): return self.num

    def _which(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.num:
            raise IndexError(f"index {idx} out of range [0,{self.num})")
        sid = bisect_right(self._ends, idx)  # first end > idx
        off = idx - self._offsets[sid]
        return sid, off

    def get_by_sid_off(self, sid: int, off: int) -> np.ndarray:
        mm = self._ensure_open(sid)
        return mm[off]

    def _ensure_open(self, sid: int):
        if sid in self._open:
            # refresh LRU
            try:
                self._lru.remove(sid)
            except ValueError:
                pass
            self._lru.append(sid)
            return self._open[sid]
        mm = np.load(self._paths[sid], mmap_mode='r')
        self._open[sid] = mm
        self._lru.append(sid)
        while len(self._lru) > self.cache:
            ev = self._lru.pop(0)
            try:
                del self._open[ev]
            except Exception:
                pass
        return mm

    def get(self, idx: int) -> np.ndarray:
        sid, off = self._which(idx)
        mm = self._ensure_open(sid)
        return mm[off]  # np.memmap view (no copy)


# --------------------------
# Dataset
# --------------------------

class CustomDataset(Dataset):
    """
    Returns:
      if load_dinov3 == False:
        (raw_or_placeholder[?], vae_latents[np], label[int64])

      if load_dinov3 == True and load_registers == False:
        (raw_or_placeholder[?], vae_latents[np], label[int64],
         dinov3_patches[np], dinov3_cls[np])

      if load_dinov3 == True and load_registers == True:
        (raw_or_placeholder[?], vae_latents[np], label[int64],
         dinov3_patches[np], dinov3_cls[np], dinov3_registers[np])
    """
    def __init__(
        self,
        data_dir: str,
        load_dinov3: bool = True,
        dinov3_subdir: Optional[str] = "dinov3-vit7b16",
        load_registers: bool = False,
        load_image: bool = False,                 # if False, returns a tiny uint8 placeholder scalar
        image_cache_shards: int = 2,              # LRU cache size for image shard mmaps
        vae_cache_shards: int = 16,               # ideal default: match/≥ your window size
        dino_cache_shards: int = 16,              # ideal default: match/≥ your window size
    ):
        super().__init__()
        self.data_dir = data_dir
        self.load_dinov3 = load_dinov3
        self.load_registers = load_registers
        self.load_image = load_image

        # labels
        labels_path = os.path.join(data_dir, 'images', 'packed', 'labels.npy')
        if not os.path.exists(labels_path):
            raise FileNotFoundError("labels.npy not found under images/packed. Run preprocessing first.")
        self.labels = np.load(labels_path).astype(np.int64)

        # VAE latents
        vae_root = os.path.join(data_dir, 'vae-sd', 'packed')
        vae_meta = self._load_json(os.path.join(vae_root, 'meta.json'))
        self.vae_arr = _ShardedArray(_ShardSpec(vae_root, None, vae_meta), cache=vae_cache_shards)

        # Optional images
        self.img_arr = None
        if self.load_image:
            img_root = os.path.join(data_dir, 'images', 'packed')
            img_meta = self._load_json(os.path.join(img_root, 'meta.json'))
            self.img_arr = _ShardedArray(_ShardSpec(img_root, None, img_meta), cache=image_cache_shards)

        # DINOv3
        if load_dinov3:
            dino_root = os.path.join(data_dir, dinov3_subdir, 'packed')
            dino_meta = self._load_json(os.path.join(dino_root, 'meta.json'))
            self.dino_cls  = _ShardedArray(_ShardSpec(dino_root, 'cls',     dino_meta), cache=dino_cache_shards)
            self.dino_pat  = _ShardedArray(_ShardSpec(dino_root, 'patches', dino_meta), cache=dino_cache_shards)
            self.dino_regs = _ShardedArray(_ShardSpec(dino_root, 'registers', dino_meta), cache=dino_cache_shards) if load_registers else None

            n = len(self.labels)
            assert len(self.vae_arr) == n == len(self.dino_cls) == len(self.dino_pat), "Shard lengths mismatch (vae/cls/patches/labels)."
            if self.dino_regs is not None:
                assert len(self.dino_regs) == n, "Shard lengths mismatch (registers)."
        else:
            assert len(self.vae_arr) == len(self.labels), "VAE/labels mismatch."

        self._shared_layout = True
        try:
            ref_counts = self.vae_arr._counts
            for arr in [self.img_arr, getattr(self, "dino_cls", None),
                        getattr(self, "dino_pat", None), getattr(self, "dino_regs", None)]:
                if arr is not None and arr._counts != ref_counts:
                    self._shared_layout = False
                    break
        except Exception:
            self._shared_layout = False

    def _load_json(self, p: str) -> dict:
        with open(p,'r') as f: return json.load(f)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx: int):
        lab = int(self.labels[idx])

        if self._shared_layout:
            sid, off = self.vae_arr._which(idx)
            vae = self.vae_arr.get_by_sid_off(sid, off)
            if self.img_arr is not None:
                img = self.img_arr.get_by_sid_off(sid, off)
            else:
                img = np.uint8(1)
            if not self.load_dinov3:
                return (img, vae, lab)
            cls = self.dino_cls.get_by_sid_off(sid, off)
            pat = self.dino_pat.get_by_sid_off(sid, off)
            if self.dino_regs is None:
                return (img, vae, lab, pat, cls)
            regs = self.dino_regs.get_by_sid_off(sid, off)
            return (img, vae, lab, pat, cls, regs)

        # Fallback: independent lookups (still O(log S) now that _which is fixed)
        vae = self.vae_arr.get(idx)
        if self.img_arr is not None:
            img = self.img_arr.get(idx)
        else:
            img = np.uint8(1)
        if not self.load_dinov3:
            return (img, vae, lab)
        cls = self.dino_cls.get(idx)
        pat = self.dino_pat.get(idx)
        if self.dino_regs is None:
            return (img, vae, lab, pat, cls)
        regs = self.dino_regs.get(idx)
        return (img, vae, lab, pat, cls, regs)

    # Expose shard info for samplers
    def shard_info(self) -> Dict[str,Any]:
        S = self.vae_arr
        return {"num_shards": len(S._paths), "records_per_shard": S.rps, "counts": S._counts[:], "offsets": S._offsets[:]}


# --------------------------
# DDP-aware shard-window sampler
# --------------------------

class ShardWindowSampler(Sampler[int]):
    """
    Keeps I/O hot by sampling only from a window of shards each epoch, while
    still providing randomness. DDP-aware: slices indices by (rank, world).

    Randomizes *across* chunks but reads *sequentially inside* each chunk to help OS readahead.
    Use set_epoch(epoch) every epoch for fresh order.
    """
    def __init__(self, dataset: CustomDataset, window_shards: int = 16, strategy: str = 'cover',
                 samples_per_epoch: Optional[int] = None, seed: int = 2025, drop_last: bool = True,
                 ddp_rank: int = 0, ddp_world: int = 1, chunk_size: Optional[int] = None):
        info = dataset.shard_info()
        self.ds = dataset
        self.num_shards = info["num_shards"]
        self.counts = info["counts"]
        self.offsets = info["offsets"]
        self.rps = info["records_per_shard"]
        self.window_shards = max(1, min(window_shards, self.num_shards))
        self.strategy = strategy
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._cover = np.arange(self.num_shards)
        self.rank = int(ddp_rank)
        self.world = int(ddp_world)
        # default chunk: at least a shard's RPS, or 1024 samples — whichever larger
        self.chunk = int(chunk_size) if chunk_size is not None else max(4096, 2 * self.rps)


    def set_epoch(self, epoch: int): self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if self.strategy == 'cover':
            rng_cov = np.random.default_rng(self.seed)
            if self.epoch == 0:
                rng_cov.shuffle(self._cover)
            w = self.window_shards
            # rotate window each epoch
            start = (self.epoch * w) % max(w, self.num_shards)
            shard_ids = self._cover[start:start+w].tolist()
            if len(shard_ids) < w:
                shard_ids += self._cover[:(w-len(shard_ids))].tolist()
        else:
            rng_cov = np.random.default_rng(self.seed + self.epoch)
            shard_ids = rng_cov.choice(self.num_shards, size=self.window_shards, replace=False).tolist()

        rng = np.random.default_rng(self.seed + self.epoch)
        idxs: List[int] = []
        chunk = self.chunk

        # Build randomized chunk order per shard, but sequential within each chunk
        for sid in shard_ids:
            start = self.offsets[sid]; end = start + self.counts[sid]
            starts = list(range(start, end, chunk))
            rng.shuffle(starts)
            for s in starts:
                e = min(s + chunk, end)
                idxs.extend(range(s, e))

        # DDP slice
        idxs = idxs[self.rank::self.world]

        if self.samples_per_epoch is not None:
            idxs = idxs[:self.samples_per_epoch]
        return iter(idxs)

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch // max(1,self.world)
        # approximate: sum counts of first window_shards shards
        total = sum(self.counts[:self.window_shards])
        return total // max(1,self.world)


# --------------------------
# DDP test harness
# --------------------------

def _ddp_env():
    is_dist = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_dist:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        dist.init_process_group(backend=backend, init_method="env://")
    return is_dist, rank, world, local_rank

def _allreduce_sum(x: float) -> float:
    if not torch.distributed.is_initialized(): return x
    t = torch.tensor([x], dtype=torch.float64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t.item())

def _allreduce_max(x: float) -> float:
    if not torch.distributed.is_initialized(): return x
    t = torch.tensor([x], dtype=torch.float64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return float(t.item())

def main():
    parser = argparse.ArgumentParser("Dataset DDP pass-through benchmark")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dinov3-subdir", type=str, default="dinov3-vit7b16")
    parser.add_argument("--no-dinov3", action="store_true")
    parser.add_argument("--no-registers", action="store_true")
    parser.add_argument("--load-image", action="store_true")  # default False

    parser.add_argument("--batch-size", type=int, default=256)

    # Ideal defaults
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8)

    # pin_memory & persistent_workers default ON, with opt-outs
    pm_group = parser.add_mutually_exclusive_group()
    pm_group.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    pm_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)

    pw_group = parser.add_mutually_exclusive_group()
    pw_group.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    pw_group.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    parser.add_argument("--limit", type=int, default=0, help="Number of batches (per rank) to iterate; 0 = all in window")
    parser.add_argument("--sampler", type=str, choices=["distributed","shardwindow"], default="shardwindow")
    parser.add_argument("--window-shards", type=int, default=16)
    parser.add_argument("--strategy", type=str, choices=["cover","random"], default="cover")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--chunk-size", type=int, default=0, help="Override sampler chunk size; 0 = auto (max(1024,RPS))")
    args = parser.parse_args()

    is_dist, rank, world, local_rank = _ddp_env()

    ds = CustomDataset(
        data_dir=args.data_dir,
        load_dinov3=(not args.no_dinov3),
        dinov3_subdir=args.dinov3_subdir,
        load_registers=(not args.no_registers),
        load_image=args.load_image,
        # match caches to window for minimal shard open/close thrash
        image_cache_shards=max(2, args.window_shards if args.load_image else 2),
        vae_cache_shards=max(4, args.window_shards),
        dino_cache_shards=max(4, args.window_shards),
    )

    if args.sampler == "shardwindow":
        sampler = ShardWindowSampler(
            ds,
            window_shards=args.window_shards,
            strategy=args.strategy,
            samples_per_epoch=None,
            seed=args.seed,
            drop_last=False,
            ddp_rank=rank,
            ddp_world=world,
            chunk_size=(args.chunk_size if args.chunk_size > 0 else None),
        )
    else:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(
            ds,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            drop_last=True,
        ) if is_dist else None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=args.persistent_workers and (args.num_workers > 0),
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    if rank == 0:
        print(
            f"[INFO] world={world} | bs/global={args.batch_size*world} | workers/replica={args.num_workers} "
            f"| sampler={args.sampler} | window={args.window_shards} | chunk={('auto' if args.chunk_size==0 else args.chunk_size)} "
            f"| load_image={args.load_image} | load_registers={not args.no_registers} "
            f"| pin_memory={args.pin_memory} | persistent_workers={args.persistent_workers}"
        )

    if args.sampler == "shardwindow" and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    seen_local = 0
    steps = 0

    # Show progress only on rank 0
    iterator = tqdm(dl, total=len(dl), disable=(rank != 0), desc="Epoch 0", dynamic_ncols=True)

    for batch in iterator:
        # batch is a tuple of Tensors (DataLoader converts np arrays for you)
        # Index 1 is VAE latents in both return modes
        bsz = batch[1].shape[0]
        steps += 1
        seen_local += bsz

        if rank == 0:
            elapsed = max(1e-6, time.perf_counter() - t0)
            local_ips = seen_local / elapsed
            iterator.set_postfix_str(f"local_ips={local_ips:.1f}")

        if args.limit > 0 and steps >= args.limit:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_local = time.perf_counter() - t0

    total_samples = _allreduce_sum(seen_local) if is_dist else seen_local
    slowest_time = _allreduce_max(dt_local) if is_dist else dt_local
    if rank == 0:
        ips = total_samples / max(1e-6, slowest_time)
        print(f"[RESULT] samples={int(total_samples)}  time={slowest_time:.3f}s  throughput={ips:.2f} samples/s (global)")

    if is_dist:
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
