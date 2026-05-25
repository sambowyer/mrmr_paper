#!/usr/bin/env python3
"""Precompute per-question relevance MI for each unique (dataset, training split).

For every unique (dataset, train_model_indices) combination found in the
results tree, computes MI(question_j, target) for ALL questions j using the
Ross estimator (binary) or KSG (continuous), and saves the relevance vector
to a cache file.

Per-coreset redundancy is NOT precomputed here — it is cheap to compute
on-the-fly (O(K^2 * M) for binary data) and the Streamlit app does so by
indexing into the score matrix.

Cache files are saved to ``mi_cache/{dataset}/{hash}.npz`` where the hash
uniquely identifies the train_model_indices tuple.  Each file contains:

  - ``relevance``: float32 array of shape (N_questions,)
  - ``train_model_indices``: int32 array of shape (M_train,)
  - ``binary``: bool scalar — whether the score matrix is binary

Already-computed cache files are skipped, making the script safe to re-run.

Usage:
    python precompute_mi_cache.py [--results-dir ./results] [--dry-run]
    python precompute_mi_cache.py --dataset arc_challenge --workers 8
"""

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib as jbl
import numpy as np
from scipy.special import psi
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# MI estimators (inlined to avoid torch import via benchpred)
# ---------------------------------------------------------------------------

def _is_binary_scores(scores: np.ndarray) -> bool:
    finite = scores[np.isfinite(scores)]
    return finite.size > 0 and len(np.unique(finite)) <= 2


def _mi_ross(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Ross (2014) MI estimator for binary x and continuous y."""
    n = len(x)
    if n == 0:
        return 0.0
    y0 = y[x == 0].reshape(-1, 1)
    y1 = y[x == 1].reshape(-1, 1)
    nx0, nx1 = len(y0), len(y1)
    if nx0 < k + 1 or nx1 < k + 1:
        return 0.0
    tree0 = cKDTree(y0)
    tree1 = cKDTree(y1)
    tree_all = cKDTree(y.reshape(-1, 1))
    m_list = []
    for i in range(n):
        val = y[i].reshape(1, -1)
        tree_same = tree1 if x[i] == 1 else tree0
        dist, _ = tree_same.query(val, k=k + 1)
        max_dist = np.max(dist)
        m = tree_all.query_ball_point(
            val.reshape(1, -1), max_dist - 1e-15, return_length=True
        )
        m_list.append(m)
    avg_psi_nx = (nx0 * psi(nx0) + nx1 * psi(nx1)) / n
    avg_psi_m = np.mean([psi(m) for m in m_list])
    return max(0.0, psi(n) - avg_psi_nx + psi(k) - avg_psi_m)


def _mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """KSG1 MI estimator for two continuous variables."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < k + 1:
        return 0.0
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))
    dd, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dd[:, -1]
    r = np.maximum(eps - 1e-15, 0.0)
    nx = tree_x.query_ball_point(
        x.reshape(-1, 1), r, p=np.inf, return_length=True
    ) - 1
    ny = tree_y.query_ball_point(
        y.reshape(-1, 1), r, p=np.inf, return_length=True
    ) - 1
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    mi = psi(k) + psi(n) - np.mean(psi(nx + 1) + psi(ny + 1))
    return max(0.0, float(mi))


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

def mi_cache_hash(train_model_indices) -> str:
    """Deterministic 16-char hex hash of a train_model_indices sequence."""
    arr = np.array(sorted(train_model_indices), dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def mi_cache_path(cache_dir: str, dataset: str, train_model_indices) -> str:
    h = mi_cache_hash(train_model_indices)
    return os.path.join(cache_dir, dataset, f"{h}.npz")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_relevance_all(
    source_full_scores: np.ndarray,
    binary: bool,
) -> np.ndarray:
    """MI(question_j, target) for every question j."""
    num_data = source_full_scores.shape[1]
    target = source_full_scores.mean(axis=1)
    mi_rel = _mi_ross if binary else _mi_ksg
    relevance = np.empty(num_data, dtype=np.float32)
    for j in range(num_data):
        relevance[j] = mi_rel(source_full_scores[:, j], target, k=3)
    return relevance


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

_scores_cache: dict[str, np.ndarray] = {}


def _load_scores(scores_path: str) -> np.ndarray:
    if scores_path not in _scores_cache:
        _scores_cache[scores_path] = jbl.load(scores_path)
    return _scores_cache[scores_path]


# ---------------------------------------------------------------------------
# Group discovery
# ---------------------------------------------------------------------------

def _discover_groups(
    results_root: str,
    data_dir: str,
    dataset_filter: str | None,
    only_discrete: bool,
) -> dict[tuple[str, tuple[int, ...]], None]:
    """Find all unique (dataset, train_model_indices) groups in results/.

    Returns a dict whose keys are (dataset, tuple(train_model_indices)).
    """
    groups: dict[tuple[str, tuple[int, ...]], None] = {}
    seen_datasets: dict[str, bool] = {}

    for split_method in sorted(os.listdir(results_root)):
        sm_path = os.path.join(results_root, split_method)
        if not os.path.isdir(sm_path):
            continue
        for cs_dir in sorted(os.listdir(sm_path)):
            cs_path = os.path.join(sm_path, cs_dir)
            if not os.path.isdir(cs_path):
                continue
            for nm_dir in sorted(os.listdir(cs_path)):
                nm_path = os.path.join(cs_path, nm_dir)
                if not os.path.isdir(nm_path):
                    continue
                for dataset in sorted(os.listdir(nm_path)):
                    dataset_path = os.path.join(nm_path, dataset)
                    if not os.path.isdir(dataset_path) or dataset.startswith("_"):
                        continue
                    if dataset_filter and dataset != dataset_filter:
                        continue

                    scores_path = os.path.join(data_dir, f"{dataset}.jbl")
                    if not os.path.exists(scores_path):
                        continue

                    if dataset not in seen_datasets:
                        scores = _load_scores(scores_path)
                        seen_datasets[dataset] = _is_binary_scores(scores)

                    if only_discrete and not seen_datasets[dataset]:
                        continue

                    for method in os.listdir(dataset_path):
                        method_path = os.path.join(dataset_path, method)
                        if not os.path.isdir(method_path):
                            continue
                        for seed_dir in os.listdir(method_path):
                            seed_path = os.path.join(method_path, seed_dir)
                            rp = os.path.join(seed_path, "result.jbl")
                            if not os.path.isfile(rp):
                                continue
                            try:
                                result = jbl.load(rp)
                            except Exception:
                                continue
                            tmi = result.get("train_model_indices")
                            if tmi is not None:
                                key = (dataset, tuple(tmi))
                                groups[key] = None

    return groups


# ---------------------------------------------------------------------------
# Per-group worker
# ---------------------------------------------------------------------------

def _process_group(
    dataset: str,
    train_indices: tuple[int, ...],
    data_dir: str,
    cache_dir: str,
    dry_run: bool,
    force: bool,
) -> str:
    """Compute and save the MI cache for one (dataset, train_indices) group."""
    out_path = mi_cache_path(cache_dir, dataset, train_indices)

    if not force and os.path.exists(out_path):
        return "skip_exists"

    if dry_run:
        return "would_write"

    scores_path = os.path.join(data_dir, f"{dataset}.jbl")
    scores = _load_scores(scores_path)
    binary = _is_binary_scores(scores)

    train_idx = np.asarray(train_indices, dtype=np.int32)
    if train_idx.max() >= scores.shape[0]:
        return "skip_index_oob"

    source_full_scores = scores[train_idx]

    try:
        relevance = _compute_relevance_all(source_full_scores, binary)
    except Exception:
        return "skip_compute_error"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        relevance=relevance,
        train_model_indices=train_idx,
        binary=np.array([binary]),
    )
    return "written"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="Root results directory (default: ./results)",
    )
    parser.add_argument(
        "--data-dir", default="./data/scores",
        help="Directory containing {dataset}.jbl score files (default: ./data/scores)",
    )
    parser.add_argument(
        "--cache-dir", default="./mi_cache",
        help="Output directory for MI cache files (default: ./mi_cache)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing anything.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite existing cache files.",
    )
    parser.add_argument(
        "--only-discrete", action="store_true", default=True,
        help="Skip datasets with continuous scores (default: True).",
    )
    parser.add_argument(
        "--include-continuous", action="store_true",
        help="Also process datasets with continuous scores.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Only process this dataset name (e.g. arc_challenge).",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8).",
    )
    args = parser.parse_args()

    if args.include_continuous:
        args.only_discrete = False

    results_root = os.path.abspath(args.results_dir)
    data_dir = os.path.abspath(args.data_dir)
    cache_dir = os.path.abspath(args.cache_dir)

    if not os.path.isdir(results_root):
        print(f"ERROR: {results_root} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print("Phase 1: discovering unique (dataset, train_split) groups ...")
    t0 = time.time()
    groups = _discover_groups(results_root, data_dir, args.dataset, args.only_discrete)
    t1 = time.time()
    print(f"  Found {len(groups)} unique groups ({t1 - t0:.1f}s)")

    ds_counts = defaultdict(int)
    for (ds, _) in groups:
        ds_counts[ds] += 1
    for ds in sorted(ds_counts):
        print(f"    {ds}: {ds_counts[ds]} groups")

    print(f"\n  Settings: cache_dir={cache_dir}, only_discrete={args.only_discrete}, "
          f"force={args.force}, dry_run={args.dry_run}")
    print()

    if not groups:
        print("Nothing to process.")
        return

    print(f"Phase 2: computing relevance vectors ({args.workers} workers) ...")
    counts: dict[str, int] = defaultdict(int)
    done = 0
    total = len(groups)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for (dataset, train_indices) in groups:
            fut = executor.submit(
                _process_group,
                dataset, train_indices, data_dir, cache_dir,
                args.dry_run, args.force,
            )
            futures[fut] = dataset

        for future in as_completed(futures):
            try:
                status = future.result()
            except Exception as exc:
                ds = futures[future]
                print(f"  ERROR ({ds}): {exc}", file=sys.stderr)
                status = "skip_compute_error"
            counts[status] += 1
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t1
                print(f"  [{done}/{total}, {elapsed:.0f}s] {dict(counts)}")

    print(f"\nDone. Summary: {dict(counts)}")
    if args.dry_run:
        print("(dry-run mode — no files were written)")


if __name__ == "__main__":
    main()
