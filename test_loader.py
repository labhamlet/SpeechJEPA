#!/usr/bin/env python3
"""
verify_ssl_training.py

Three independent checks for multi-node / multi-worker WebDataset SSL training.

  cuda   (point 2) Does a top-level torch.cuda.empty_cache() (as in your
                   cleanup_memory()) actually create a CUDA context, and does
                   every rank land on its OWN physical GPU (no piling on GPU 0)?
  shards (point 3) Are DataLoader workers FORKED, is the process group visible
                   INSIDE workers, does resampled+split give every (rank,worker)
                   a DISTINCT shard stream, are shards PRE-SHUFFLED (many speakers
                   per shard), how DIVERSE is a batch end-to-end, and what is the
                   DUPLICATE rate within/across ranks?
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


SHARD_TEST_SEED = 1234


def _seed_like_training(seed):
    """Same base seed on EVERY rank, mirroring seed_everything(seed, workers=True)."""
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
        return
    except Exception:  # noqa: BLE001
        import random
        random.seed(seed)
        torch.manual_seed(seed)


def _make_worker_init(rank):
    """Rank-aware worker seeding matching Lightning's pl_worker_init_function, so
    workers on different ranks get different RNG state from one shared base seed."""
    def _init(worker_id):
        for modpath in ("lightning_fabric.utilities.seed",
                        "pytorch_lightning.utilities.seed"):
            try:
                mod = __import__(modpath, fromlist=["pl_worker_init_function"])
                fn = getattr(mod, "pl_worker_init_function")
                try:
                    fn(worker_id, rank)
                except TypeError:
                    fn(worker_id)               # older signature reads RANK from env
                return
            except Exception:  # noqa: BLE001
                continue
        torch.manual_seed((torch.initial_seed() % (2 ** 31)) + 100003 * rank + worker_id)
    return _init


def _splitters():
    from webdataset.shardlists import split_by_node, split_by_worker
    return split_by_node, split_by_worker


class _ShardDrawProbe(torch.utils.data.IterableDataset):
    """Taps WebDataset's raw shard SAMPLING stream (no tar I/O).

    Reading actual samples makes a worker stream one whole shard before the
    next, so a short probe makes every worker look like it saw ONE shard.
    Sampling shard URLs directly tests the per-(rank, worker) RNG independently
    of shard size.

    When mirror_split=True the stream is passed through split_by_node ->
    split_by_worker, i.e. the EXACT distribution your training pipeline uses
    once the nodesplitter is restored. That is the config we actually ship, so
    that is what the cross-rank check should exercise.
    """

    def __init__(self, shards, n_draws, mirror_split=True):
        self.shards, self.n_draws, self.mirror_split = shards, n_draws, mirror_split

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
        src = ResampledShards(_expand(self.shards))
        if self.mirror_split:
            split_by_node, split_by_worker = _splitters()
            src = split_by_node(src)        # reads dist.get_rank(); no-op if world==1
            src = split_by_worker(src)      # reads worker_info; no-op if 1 worker
        for k, d in enumerate(src):
            if k >= self.n_draws:
                break
            yield {**meta, "url": d["url"]}


def _speaker(key):
    """LibriSpeech __key__ is '<speaker>-<chapter>-<utt>'. Falls back to the
    whole key if it has no '-'."""
    return key.split("-", 1)[0]


def _read_order_probe(shards):
    """Reads ACTUAL samples (no decode, NO shuffle) to characterize the raw
    shard layout: how long the stream stays inside one shard, and -- the part
    that verifies the OFFLINE reshuffle -- how many distinct speakers live
    inside a single shard. Pre-shuffled shards => many speakers per shard;
    speaker-sorted shards => 1-2.

    Passes nodesplitter=split_by_node so it does not trip the multi-node guard.
    """
    import webdataset as wds
    split_by_node, _ = _splitters()

    def tag(s):
        wi = torch.utils.data.get_worker_info()
        return {"worker": wi.id if wi else -1,
                "url": s.get("__url__", "?"),
                "key": s.get("__key__", "?")}

    return wds.WebDataset(shards, resampled=True, shardshuffle=False,
                          nodesplitter=split_by_node,
                          handler=wds.warn_and_continue).map(tag)


def _faithful_key_probe(shards, shuffle_buffer):
    """Mirrors the TRAINING pipeline through the shuffle stage, reading only
    __key__ (no audio decode). Construction matches make_web_dataset:
    resampled + split_by_node + split_by_worker + .shuffle(N).

    We map to {key,url,worker} BEFORE .shuffle so the buffer stays light --
    the emitted KEY ORDER is identical either way (shuffle permutes by buffer
    position, not by payload), so this is faithful for measuring batch
    composition while costing a fraction of the RAM.
    """
    import webdataset as wds
    split_by_node, split_by_worker = _splitters()

    def to_meta(s):
        wi = torch.utils.data.get_worker_info()
        return {"key": s.get("__key__", "?"), "url": s.get("__url__", "?"),
                "worker": wi.id if wi else -1}

    return (
        wds.WebDataset(shards, resampled=True, shardshuffle=False,
                       nodesplitter=split_by_node, workersplitter=split_by_worker,
                       handler=wds.warn_and_continue)
        .map(to_meta)
        .shuffle(shuffle_buffer)
    )


def check_shards(rank, world, pg, shards, num_workers, draws_per_worker, order_samples,
                 shuffle_buffer, batch_utts, div_samples):
    if shards is None:
        print(f"[shards][rank {rank}] --shards not provided; skipping.", flush=True)
        return

    all_urls = _expand(shards)
    if rank == 0:
        print(f"[shards] --shards expands to {len(all_urls)} shards "
              f"(first={all_urls[0]}, last={all_urls[-1]}). "
              f"If this count is way too low, your pattern is wrong.", flush=True)

    # ---- Part A: SEEDING via raw shard draws (shard-size independent) -------
    # Reproduce training's seeding EXACTLY, or this test passes for the wrong
    # reason: seed_everything gives every rank the SAME base seed, and Lightning's
    # worker_init folds the rank back in. If we skipped this, each process would
    # get a random seed and ranks would diverge trivially -- hiding real bugs.
    _seed_like_training(SHARD_TEST_SEED)
    probe = _ShardDrawProbe(shards, draws_per_worker, mirror_split=True)
    loader = DataLoader(probe, batch_size=None, num_workers=num_workers,
                        prefetch_factor=(2 if num_workers > 0 else None),
                        worker_init_fn=_make_worker_init(rank))
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


    order = _read_order_probe(shards)
    oloader = DataLoader(order, batch_size=None, num_workers=1, prefetch_factor=2)
    last, run, max_run, switches, total = None, 0, 0, 0, 0
    first_shard_url, first_shard_spk = None, set()
    for s in oloader:
        u, k = s["url"], s["key"]
        if first_shard_url is None:
            first_shard_url = u
        if u == first_shard_url:
            first_shard_spk.add(_speaker(k))
        if u == last:
            run += 1
        else:
            max_run = max(max_run, run)
            run, last, switches = 1, u, switches + 1
        total += 1
        if total >= order_samples:
            break
    max_run = max(max_run, run)
    verdict = ("GOOD: shards are pre-shuffled" if len(first_shard_spk) >= 50
               else "BAD: shards are speaker-sorted -- rebuild with a global shuffle")
    print(f"[shards][rank {rank}] WITHIN-SHARD: longest single-shard run={max_run}, "
          f"shard switches={switches}; first shard holds {len(first_shard_spk)} distinct "
          f"speakers. {verdict}.", flush=True)

    _seed_like_training(SHARD_TEST_SEED)
    fk = _faithful_key_probe(shards, shuffle_buffer)
    floader = DataLoader(fk, batch_size=None, num_workers=num_workers,
                         prefetch_factor=(2 if num_workers > 0 else None),
                         worker_init_fn=_make_worker_init(rank))
    keys_by_worker = defaultdict(list)
    budget = num_workers * div_samples if num_workers > 0 else div_samples
    for n, s in enumerate(floader):
        keys_by_worker[s["worker"]].append(s["key"])
        if n + 1 >= budget:
            break

    # C1 -- batch diversity: distinct speakers per window of `batch_utts`
    ratios, ex_distinct = [], None
    for w, keys in keys_by_worker.items():
        for i in range(0, len(keys) - batch_utts + 1, batch_utts):
            window = keys[i:i + batch_utts]
            d = len({_speaker(k) for k in window})
            ratios.append(d / batch_utts)
            if ex_distinct is None:
                ex_distinct = d
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    print(f"[shards][rank {rank}] DIVERSITY: a batch of ~{batch_utts} utts holds "
          f"~{mean_ratio * batch_utts:.1f} distinct speakers on average "
          f"(ratio {mean_ratio:.2f}; 1.00=every utt a different speaker). "
          f"Low ratio => shrink shards or raise --shuffle-buffer.", flush=True)

    # C2 -- duplicates within this rank
    all_keys = [k for ks in keys_by_worker.values() for k in ks]
    n_keys = len(all_keys)
    intra = sum(len(ks) - len(set(ks)) for ks in keys_by_worker.values())
    seen_in = defaultdict(set)
    for w, ks in keys_by_worker.items():
        for k in set(ks):
            seen_in[k].add(w)
    cross_worker = sum(1 for k, ws in seen_in.items() if len(ws) > 1)
    print(f"[shards][rank {rank}] DUPLICATES (within rank, {n_keys} draws): "
          f"intra-worker repeats={intra} ({100 * intra / max(n_keys, 1):.1f}%), "
          f"utts seen by >1 worker={cross_worker}. "
          f"Low single-digit % is NORMAL for resampled=True (sampling with "
          f"replacement); near-total overlap would mean identical streams.",
          flush=True)

    # C2 -- duplicates ACROSS ranks (the one that wastes DDP compute if broken)
    if pg:
        my_set = set(all_keys)
        gathered = [None] * world
        dist.all_gather_object(gathered, {"rank": rank, "keys": list(my_set),
                                           "n": len(my_set)})
        if rank == 0:
            from collections import Counter as _C
            rank_count = _C()
            for g in gathered:
                for k in g["keys"]:
                    rank_count[k] += 1
            shared = sum(1 for k, c in rank_count.items() if c > 1)
            total_distinct = len(rank_count)
            sigs = [hashlib.md5("|".join(sorted(g["keys"])).encode()).hexdigest()[:10]
                    for g in gathered]
            distinct_streams = len(set(sigs)) == world
            print(f"[shards][SUMMARY] CROSS-RANK utterances: {shared}/{total_distinct} "
                  f"({100 * shared / max(total_distinct, 1):.1f}%) appear on >1 rank; "
                  f"per-rank key-streams distinct={distinct_streams}. "
                  f"Single-digit %% + distinct=True is healthy resampling; "
                  f"~100%% would mean ranks process the SAME data (wasted compute).",
                  flush=True)


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
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--draws-per-worker", type=int, default=500,
                    help="shard URLs sampled per worker for the seeding test")
    ap.add_argument("--order-samples", type=int, default=3000,
                    help="consecutive samples read to measure within-shard diversity")
    ap.add_argument("--shuffle-buffer", type=int, default=2000,
                    help="runtime .shuffle(N) buffer to mirror (match your data module)")
    ap.add_argument("--batch-utts", type=int, default=128,
                    help="approx utterances per batch, for the diversity window")
    ap.add_argument("--div-samples", type=int, default=4000,
                    help="post-shuffle samples per worker for diversity/duplicate stats")
    # RAM knobs (defaults mirror your config)
    ap.add_argument("--target-numel", type=int, default=6_000_000)
    ap.add_argument("--max-numel", type=int, default=6_000_000)
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
                     args.draws_per_worker, args.order_samples,
                     args.shuffle_buffer, args.batch_utts, args.div_samples)
    if "ram" in checks and rank == 0:        # single-process; run once on rank 0
        check_ram(args)

    if pg:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()