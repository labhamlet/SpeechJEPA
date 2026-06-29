#!/usr/bin/env python3
"""
verify_ssl_training.py

Three independent checks for multi-node / multi-worker WebDataset SSL training.

  cuda   (point 2) Does a top-level torch.cuda.empty_cache() (as in your
                   cleanup_memory()) actually create a CUDA context, and does
                   every rank land on its OWN physical GPU (no piling on GPU 0)?
  shards (point 3) Are DataLoader workers FORKED (not spawned), is the process
                   group visible INSIDE workers, and does resampled=True give
                   every (rank, worker) a DISTINCT shard stream (no cross-rank
                   or cross-worker duplication)?
  ram              Does bucket_limits=True actually collapse padded-tensor shape
                   variety (-> allocator fragmentation + torch.compile recompiles)
                   and lower peak host RSS, vs bucket_limits=False -- and what
                   does it cost in extra padding compute?

Distributed checks -- run under your REAL launcher so the env matches training:

  srun -N2 --ntasks-per-node=2 --gpus-per-task=1 --cpus-per-task=8 \
       python verify_ssl_training.py --checks cuda,shards \
       --shards "/shared/data/train-{000000..001234}.tar" \
       --rendezvous-dir /shared/fs/tmp --num-workers 8 --samples 3000

  # --shards must be the SAME brace pattern / list you pass in training.
  # --rendezvous-dir must be on a filesystem visible to ALL nodes.

RAM check -- single process, no GPU, no shards needed:

  python verify_ssl_training.py --checks ram
"""
import argparse
import datetime
import gc
import hashlib
import os
import resource
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Mirrors SSLDataModule.BUCKET_BOUNDARIES -- keep in sync with your data module.
BUCKET_BOUNDARIES = [32000, 48000, 64000, 80000, 96000, 112000, 128000,
                     144000, 160000, 176000, 192000, 208000, 224000, 250000]


# --------------------------------------------------------------------------- #
# distributed plumbing
# --------------------------------------------------------------------------- #
def get_dist_env():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    return rank, world, local


def maybe_init_pg(rank, world, rendezvous_dir):
    """Init a gloo group over a shared-FS FileStore so we can gather across ranks.

    We deliberately init the group in the PARENT before building any DataLoader:
    that is the exact condition that makes (or breaks) WebDataset's per-rank
    seeding -- a forked worker inherits this group and get_rank() works; a
    spawned worker does not.
    """
    if world < 2:
        return False
    if rendezvous_dir is None:
        if rank == 0:
            print("[warn] --rendezvous-dir not set; cross-rank comparison disabled. "
                  "Pass a shared-filesystem path to enable it.", flush=True)
        return False
    job = os.environ.get("SLURM_JOB_ID", "local")
    store_file = os.path.join(rendezvous_dir, f"verify_pg_{job}")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
    )
    return True


# --------------------------------------------------------------------------- #
# (2) CUDA context placement
# --------------------------------------------------------------------------- #
def check_cuda(rank, world, local, pg):
    pre_init = torch.cuda.is_initialized()           # should be False at script start
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    ndev = torch.cuda.device_count()

    # The suspect call from your top-level cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    post_empty = torch.cuda.is_initialized()

    # Now do what Lightning does: pick a device by local rank and allocate.
    intended = 0 if ndev == 0 else local % max(ndev, 1)
    landed = None
    if ndev > 0:
        torch.cuda.set_device(intended)
        t = torch.zeros(1024, 1024, device=f"cuda:{intended}")  # ~4 MB, forces a context
        t.fill_(1.0)
        landed = torch.cuda.current_device()

    print(f"[cuda][rank {rank}] CUDA_VISIBLE_DEVICES={visible} visible_count={ndev} "
          f"local_rank={local} intended_index={intended} landed_index={landed} | "
          f"is_initialized before_any={pre_init} after_empty_cache={post_empty}",
          flush=True)

    if post_empty and not pre_init:
        print(f"[cuda][rank {rank}] NOTE: empty_cache() CREATED a context on this build "
              f"-> top-level cleanup_memory() is NOT a no-op; move it inside main().",
              flush=True)
    else:
        print(f"[cuda][rank {rank}] empty_cache() created no context (guarded) "
              f"-> top-level cleanup_memory() is harmless here.", flush=True)

    _nvml_gpu_sharing(rank, world, pg)


def _nvml_gpu_sharing(rank, world, pg):
    """Use NVML's GLOBAL view (independent of CUDA_VISIBLE_DEVICES) to detect
    multiple ranks sharing one physical GPU -- the real symptom of mis-placement."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"[cuda] pynvml unavailable ({e}); skipping cross-rank GPU-sharing check "
                  f"(pip install nvidia-ml-py to enable).", flush=True)
        return

    if pg:
        dist.barrier()  # ensure every rank has allocated before we read NVML

    mypid = os.getpid()
    my_gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            pids = [p.pid for p in procs]
        except Exception:  # noqa: BLE001
            pids = []
        if mypid in pids:
            my_gpus.append(i)
    print(f"[cuda][rank {rank}] pid={mypid} on physical GPUs={my_gpus}", flush=True)

    if not pg:
        return
    gathered = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "gpus": my_gpus})
    if rank == 0:
        by_gpu = defaultdict(list)
        for g in gathered:
            for i in g["gpus"]:
                by_gpu[i].append(g["rank"])
        shared = {i: rs for i, rs in by_gpu.items() if len(rs) > 1}
        if shared:
            print(f"[cuda][SUMMARY] FAIL: physical GPUs shared by >1 rank: {shared} "
                  f"-> ranks are piling on the same device.", flush=True)
        else:
            print(f"[cuda][SUMMARY] OK: each rank on its own physical GPU ({dict(by_gpu)}).",
                  flush=True)


# --------------------------------------------------------------------------- #
# (3) fork + per-rank / per-worker shard divergence
# --------------------------------------------------------------------------- #
def _expand(shards):
    """Brace-expand the shard pattern the same way WebDataset does."""
    try:
        from webdataset.shardlists import expand_urls
    except Exception:  # noqa: BLE001
        from webdataset.utils import expand_urls  # older layouts
    return expand_urls(shards)


class _ShardDrawProbe(torch.utils.data.IterableDataset):
    """Taps WebDataset's raw shard SAMPLING stream (no tar I/O).

    This is the key fix over the naive probe: reading actual samples makes a
    worker stream one whole shard before the next, so with a short probe every
    worker looks like it only ever saw ONE shard. Sampling shard URLs directly
    tests the per-worker/per-rank RNG independently of how many utterances a
    shard holds.
    """

    def __init__(self, shards, n_draws):
        self.shards, self.n_draws = shards, n_draws

    def __iter__(self):
        try:
            from webdataset import ResampledShards
        except Exception:  # noqa: BLE001
            from webdataset.shardlists import ResampledShards
        wi = torch.utils.data.get_worker_info()
        di = dist.is_available() and dist.is_initialized()
        meta = {
            "worker": wi.id if wi else -1,
            "rank_in_worker": (dist.get_rank() if di else -1),
            "start_method": mp.get_start_method(allow_none=True) or "unknown",
            "dist_in_worker": di,
        }
        for k, d in enumerate(ResampledShards(_expand(self.shards))):
            if k >= self.n_draws:
                break
            yield {**meta, "url": d["url"]}


def _read_order_probe(shards):
    """Reads ACTUAL samples (no decode) to measure how long the stream stays
    inside a single shard -- i.e. how correlated consecutive batches are."""
    import webdataset as wds

    def tag(s):
        wi = torch.utils.data.get_worker_info()
        return {"worker": wi.id if wi else -1, "url": s.get("__url__", "?")}

    return wds.WebDataset(shards, resampled=True, shardshuffle=False,
                          handler=wds.warn_and_continue).map(tag)


def check_shards(rank, world, pg, shards, num_workers, draws_per_worker, order_samples):
    if shards is None:
        print(f"[shards][rank {rank}] --shards not provided; skipping.", flush=True)
        return

    all_urls = _expand(shards)
    if rank == 0:
        print(f"[shards] --shards expands to {len(all_urls)} shards "
              f"(first={all_urls[0]}, last={all_urls[-1]}). "
              f"If this count is way too low, your pattern is wrong.", flush=True)

    # ---- Part A: SEEDING via raw shard draws (shard-size independent) -------
    probe = _ShardDrawProbe(shards, draws_per_worker)
    loader = DataLoader(probe, batch_size=None, num_workers=num_workers,
                        prefetch_factor=(2 if num_workers > 0 else None))
    per_worker, start_methods, dist_flags, ranks_seen = defaultdict(list), set(), set(), set()
    for s in loader:
        per_worker[s["worker"]].append(s["url"])
        start_methods.add(s["start_method"])
        dist_flags.add(bool(s["dist_in_worker"]))
        ranks_seen.add(int(s["rank_in_worker"]))

    print(f"[shards][rank {rank}] worker_start_method={sorted(start_methods)} "
          f"dist_visible_in_worker={dist_flags} rank_seen_in_worker={sorted(ranks_seen)}",
          flush=True)
    if "spawn" in start_methods:
        print(f"[shards][rank {rank}] WARN: workers SPAWNED -> seeding sees no process "
              f"group; ranks may draw identical shards. Use fork.", flush=True)
    if pg and dist_flags == {False}:
        print(f"[shards][rank {rank}] WARN: process group NOT visible inside workers even "
              f"though init in the parent -> seeding falls back to rank 0.", flush=True)

    for w in sorted(per_worker):
        urls = per_worker[w]
        print(f"[shards][rank {rank}]   worker {w}: {len(set(urls))} distinct shards "
              f"/ {len(urls)} draws", flush=True)
    wh = {w: hashlib.md5("|".join(u).encode()).hexdigest()[:10] for w, u in per_worker.items()}
    cross_worker_ok = len(set(wh.values())) == len(wh)
    print(f"[shards][rank {rank}] cross_worker_seeding="
          f"{'OK' if cross_worker_ok else 'FAIL (identical shard sequences across workers!)'} "
          f"-- distinct-shards/worker should be ~min({draws_per_worker}, {len(all_urls)}).",
          flush=True)

    sig = hashlib.md5("|".join(sorted(sum(per_worker.values(), []))).encode()).hexdigest()[:12]
    if pg:
        gathered = [None] * world
        dist.all_gather_object(gathered, {"rank": rank, "sig": sig})
        if rank == 0:
            sigs = [g["sig"] for g in gathered]
            ok = len(set(sigs)) == world
            print(f"[shards][SUMMARY] cross_rank_seeding={'OK' if ok else 'FAIL'} "
                  f"({len(set(sigs))}/{world} distinct rank signatures).", flush=True)
    else:
        print(f"[shards][rank {rank}] cross-rank seeding UNTESTED (world=1). Re-run under "
              f"`srun -N>=2 --ntasks-per-node>=1` with --rendezvous-dir on shared FS.",
              flush=True)

    # ---- Part B: ORDERING -- how long does the stream stay in one shard? ----
    order = _read_order_probe(shards)
    oloader = DataLoader(order, batch_size=None, num_workers=1, prefetch_factor=2)
    last, run, max_run, switches, total = None, 0, 0, 0, 0
    for s in oloader:
        u = s["url"]
        if u == last:
            run += 1
        else:
            max_run = max(max_run, run)
            run, last, switches = 1, u, switches + 1
        total += 1
        if total >= order_samples:
            break
    max_run = max(max_run, run)
    print(f"[shards][rank {rank}] ORDER: over {total} consecutive samples (single worker), "
          f"longest single-shard run={max_run}, shard switches={switches}. "
          f"A long run => each batch is dominated by one shard (same speakers/chapters). "
          f"You have NO sample-level .shuffle() -- add one (see notes).", flush=True)


# --------------------------------------------------------------------------- #
# bucket_limits RAM re-check
# --------------------------------------------------------------------------- #
def _bucket_len(n):
    for b in BUCKET_BOUNDARIES:
        if n <= b:
            return b
    return BUCKET_BOUNDARIES[-1]


def _sample_lengths(n, lo, hi, rng):
    """Mostly-short with a long right tail (speech-like), clipped to [lo, hi]."""
    out = []
    split = lo + 0.4 * (hi - lo)
    for _ in range(n):
        x = rng.uniform(lo, split) if rng.random() < 0.9 else rng.uniform(split, hi)
        out.append(int(min(max(x, lo), hi)))
    return out


def _ram_worker(bucket_limits, q, target_numel, max_numel, buffersize, lo, hi, iters, seed):
    import random
    import psutil

    rng = random.Random(seed)
    proc = psutil.Process()
    widths, padded_sum, raw_sum, peak_kb, emitted = set(), 0, 0, 0, 0
    buf = []

    def flush():
        nonlocal emitted, padded_sum, raw_sum, peak_kb
        buf.sort()                                  # bucketing groups similar lengths
        i = 0
        while i < len(buf):
            j, cur_max = i, 0
            while j < len(buf):
                new_max = max(cur_max, buf[j])
                if (j - i + 1) * new_max > max_numel and j > i:
                    break
                cur_max = new_max
                j += 1
                if (j - i) * cur_max >= target_numel:
                    break
            batch = buf[i:j]
            B, raw_max = len(batch), max(buf[i:j])
            width = _bucket_len(raw_max) if bucket_limits else raw_max
            widths.add(width)
            raw_sum += B * raw_max
            padded_sum += B * width
            t = torch.zeros(B, width, dtype=torch.float32)  # the dominant padded tensor
            t[:, ::512] = 1.0                                # dirty ~every page
            del t
            peak_kb = max(peak_kb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            emitted += 1
            i = j
        buf.clear()

    for _ in range(iters):
        buf.extend(_sample_lengths(buffersize, lo, hi, rng))
        flush()
        if emitted > 4000:
            break

    q.put({
        "bucket_limits": bucket_limits,
        "distinct_widths": len(widths),
        "peak_rss_mb": peak_kb / 1024.0,                       # ru_maxrss is KB on Linux
        "rss_end_mb": proc.memory_info().rss / 1024.0 / 1024.0,
        "padding_overhead_pct": 100.0 * (padded_sum / raw_sum - 1.0),
        "batches": emitted,
    })


def check_ram(args):
    ctx = mp.get_context("spawn")  # fresh process per setting => clean allocator state
    res = {}
    for bl in (False, True):
        q = ctx.Queue()
        p = ctx.Process(target=_ram_worker, args=(
            bl, q, args.target_numel, args.max_numel, args.buffersize,
            args.min_sample_len, args.max_sample_len, args.ram_iters, 1234))
        p.start()
        res[bl] = q.get()
        p.join()
        r = res[bl]
        print(f"[ram] bucket_limits={bl}: distinct_padded_widths={r['distinct_widths']} "
              f"peak_rss={r['peak_rss_mb']:.0f}MB rss_end={r['rss_end_mb']:.0f}MB "
              f"padding_overhead={r['padding_overhead_pct']:.1f}% "
              f"({r['batches']} batches)", flush=True)

    f, t = res[False], res[True]
    print(f"[ram][SUMMARY] distinct padded widths: {f['distinct_widths']} -> "
          f"{t['distinct_widths']}  (fewer = less fragmentation + fewer compile recompiles).",
          flush=True)
    print(f"[ram][SUMMARY] peak RSS: {f['peak_rss_mb']:.0f}MB -> {t['peak_rss_mb']:.0f}MB; "
          f"bucketing adds ~{t['padding_overhead_pct'] - f['padding_overhead_pct']:.1f}% "
          f"padding compute.", flush=True)
    print(f"[ram][SUMMARY] The width count is the reliable, deterministic signal. RSS here is "
          f"indicative only (depends on the allocator / MALLOC_ARENA_MAX). Confirm the real "
          f"footprint during training via `sstat -j <jobid> --format=MaxRSS` over time, or the "
          f"RSSMonitor callback at the bottom of this file.", flush=True)


# Optional drop-in Lightning callback to watch host RSS during REAL training.
# Uncomment and add to your callbacks list; logs per-rank RSS to W&B.
#
# from pytorch_lightning.callbacks import Callback
# class RSSMonitor(Callback):
#     def __init__(self, every_n_steps: int = 200):
#         self.every = every_n_steps
#     def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
#         if trainer.global_step % self.every == 0:
#             import psutil
#             rss_gb = psutil.Process().memory_info().rss / 1e9
#             pl_module.log("sys/rss_gb", rss_gb, prog_bar=True, rank_zero_only=False)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="all", help="comma list of: cuda,shards,ram | all")
    ap.add_argument("--shards", default=None, help="same brace pattern/list as training")
    ap.add_argument("--rendezvous-dir", default=None, help="shared-FS dir for cross-rank gather")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--draws-per-worker", type=int, default=500,
                    help="shard URLs sampled per worker for the seeding test")
    ap.add_argument("--order-samples", type=int, default=3000,
                    help="consecutive samples read to measure single-shard run length")
    # RAM knobs (defaults mirror your config)
    ap.add_argument("--target-numel", type=int, default=6_400_000)
    ap.add_argument("--max-numel", type=int, default=6_400_000)
    ap.add_argument("--buffersize", type=int, default=8192)
    ap.add_argument("--min-sample-len", type=int, default=32000)
    ap.add_argument("--max-sample-len", type=int, default=250000)
    ap.add_argument("--ram-iters", type=int, default=200)
    args = ap.parse_args()

    checks = {"cuda", "shards", "ram"} if args.checks == "all" else set(args.checks.split(","))
    rank, world, local = get_dist_env()

    pg = False
    if checks & {"cuda", "shards"} and world > 1:
        pg = maybe_init_pg(rank, world, args.rendezvous_dir)

    if rank == 0:
        print(f"[info] world_size={world} rank={rank} local_rank={local} "
              f"process_group={'on' if pg else 'off'} checks={sorted(checks)}", flush=True)

    if "cuda" in checks:
        check_cuda(rank, world, local, pg)
    if "shards" in checks:
        check_shards(rank, world, pg, args.shards, args.num_workers,
                     args.draws_per_worker, args.order_samples)
    if "ram" in checks and rank == 0:        # single-process; run once on rank 0
        check_ram(args)

    if pg:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()