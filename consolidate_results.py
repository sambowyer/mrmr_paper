#!/usr/bin/env python3
"""Consolidate per-trial result files into Parquet for fast Streamlit loading.

Walks the results directory tree and writes three consolidated Parquet files
per ``nmodels_*`` directory, replacing hundreds of thousands of individual
file reads with a single Parquet load.

Output files (written inside each ``nmodels_*`` dir):
  _consolidated_results.parquet   -- mirrors load_all_results()
  _consolidated_crossk.parquet    -- mirrors load_cross_k_results()
  _consolidated_jbl.parquet       -- mirrors load_result_jbl_data()
  _consolidation_meta.json        -- timestamp + file counts

This script is **read-only** on all source data (.jbl, .result files).
It only writes ``_consolidated_*`` / ``_consolidation_meta.json`` files.

Re-running after new experiments does a full re-scan and overwrites the
Parquet files, correctly handling additions and deletions.

Usage:
    python consolidate_results.py                         # consolidate stale dirs
    python consolidate_results.py --nmodels-dir path/...  # single directory
    python consolidate_results.py --dry-run               # preview only
    python consolidate_results.py --force-reconsolidate   # reconsolidate all dirs
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import joblib as jbl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pass@k parsing (mirrors streamlit_app.py logic)
# ---------------------------------------------------------------------------
_PASS_AT_K_BENCHMARKS = [
    "MBPP_mbpp", "MBPPPlus_mbpp_plus",
    "HumanEvalPack_Rust", "HumanEvalPack_Java", "HumanEvalPack_Js",
    "HumanEvalPack_Go", "HumanEvalPack_Cpp", "HumanEvalPack_PythonPlus",
    "LBPP_Cpp", "LBPP_Java", "LBPP_Js",
    "LBPP_Go", "LBPP_Python", "LBPP_Rust",
]

_PASS_AT_K_RE = re.compile(
    r'^(' + '|'.join(re.escape(b) for b in _PASS_AT_K_BENCHMARKS)
    + r')_pass_at_(\d+)(_v2)?$'
)
_PRED_PASS_AT_RE = re.compile(r'^pred_pass_at_(\d+)$')


def _parse_pass_at_k_dataset_name(dataset_name: str):
    m = _PASS_AT_K_RE.match(dataset_name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3) is not None


# ---------------------------------------------------------------------------
# Record parsing (mirrors streamlit_app.py _parse_record)
# ---------------------------------------------------------------------------
def _parse_record(record_path: str) -> dict:
    result = {}
    with open(record_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "$END$":
                break
            if not line:
                continue
            try:
                result.update(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


# ---------------------------------------------------------------------------
# Collectors — one function per Parquet type
# ---------------------------------------------------------------------------
def _collect_results(nmodels_dir: str) -> list[dict]:
    """Collect rows matching load_all_results() output."""
    rows = []
    for dataset in os.listdir(nmodels_dir):
        dataset_path = os.path.join(nmodels_dir, dataset)
        if not os.path.isdir(dataset_path) or dataset.startswith("_"):
            continue
        for method in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method)
            if not os.path.isdir(method_path):
                continue
            for seed_dir in os.listdir(method_path):
                seed_path = os.path.join(method_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                record_path = os.path.join(seed_path, "record.result")
                if not os.path.exists(record_path):
                    continue
                rec = _parse_record(record_path)
                if not rec:
                    continue
                error_val = rec.get("error_MAE", rec.get("error", np.nan))
                rmse_val = rec.get("error_RMSE", np.nan)
                if np.isnan(rmse_val):
                    error_mse = rec.get("error_MSE", np.nan)
                    rmse_val = np.sqrt(error_mse) if not np.isnan(error_mse) else np.nan
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed_dir,
                    "error": error_val,
                    "rmse": rmse_val,
                    "training_time": rec.get("training_time", np.nan),
                    "inference_time": rec.get("inference_time", np.nan),
                    "corr_spearman": rec.get("corr_spearman", np.nan),
                    "corr_kendall": rec.get("corr_kendall", np.nan),
                    "corr_pearson": rec.get("corr_pearson", np.nan),
                })
    return rows


def _collect_crossk(nmodels_dir: str) -> list[dict]:
    """Collect rows matching load_cross_k_results() output."""
    rows = []
    for dataset in os.listdir(nmodels_dir):
        dataset_path = os.path.join(nmodels_dir, dataset)
        if not os.path.isdir(dataset_path) or dataset.startswith("_"):
            continue
        parsed = _parse_pass_at_k_dataset_name(dataset)
        if parsed is None:
            continue
        benchmark, source_k, _is_v2 = parsed
        for method in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method)
            if not os.path.isdir(method_path):
                continue
            for seed_dir in os.listdir(method_path):
                seed_path = os.path.join(method_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                for pred_dir in os.listdir(seed_path):
                    pm = _PRED_PASS_AT_RE.match(pred_dir)
                    if pm is None:
                        continue
                    pred_k = int(pm.group(1))
                    record_path = os.path.join(seed_path, pred_dir, "record.result")
                    if not os.path.exists(record_path):
                        continue
                    rec = _parse_record(record_path)
                    if not rec:
                        continue
                    error_val = rec.get("error_MAE", rec.get("error", np.nan))
                    rmse_val = rec.get("error_RMSE", np.nan)
                    if np.isnan(rmse_val):
                        error_mse = rec.get("error_MSE", np.nan)
                        rmse_val = (
                            np.sqrt(error_mse) if not np.isnan(error_mse)
                            else np.nan
                        )
                    rows.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed_dir,
                        "benchmark": benchmark,
                        "source_k": source_k,
                        "pred_k": pred_k,
                        "error": error_val,
                        "rmse": rmse_val,
                        "corr_spearman": rec.get("corr_spearman", np.nan),
                        "corr_kendall": rec.get("corr_kendall", np.nan),
                        "corr_pearson": rec.get("corr_pearson", np.nan),
                    })
    return rows


def _collect_jbl(nmodels_dir: str) -> list[dict]:
    """Collect rows matching load_result_jbl_data() output.

    List-valued columns are serialized as JSON strings for Parquet
    compatibility.  The Streamlit loader deserializes them back.
    """
    rows = []
    for dataset in os.listdir(nmodels_dir):
        dataset_path = os.path.join(nmodels_dir, dataset)
        if not os.path.isdir(dataset_path) or dataset.startswith("_"):
            continue
        for method in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method)
            if not os.path.isdir(method_path):
                continue
            for seed_dir in os.listdir(method_path):
                seed_path = os.path.join(method_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue
                result_path = os.path.join(seed_path, "result.jbl")
                if not os.path.exists(result_path):
                    continue
                try:
                    result = jbl.load(result_path)
                except Exception:
                    continue
                def _to_json(v):
                    if v is None:
                        return None
                    if isinstance(v, np.ndarray):
                        return json.dumps(v.tolist())
                    if isinstance(v, (list, dict)):
                        return json.dumps(v)
                    return json.dumps(v)

                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed_dir,
                    "coreset_indices": _to_json(result.get("coreset_indices")),
                    "train_model_indices": _to_json(result.get("train_model_indices")),
                    "test_model_indices": _to_json(result.get("test_model_indices")),
                    "true_acc_train": _to_json(result.get("true_acc_train")),
                    "pred_acc_train": _to_json(result.get("pred_acc_train")),
                    "true_acc_test": _to_json(result.get("true_acc_test")),
                    "pred_acc_test": _to_json(result.get("pred_acc_test")),
                    "selection_metrics": _to_json(result.get("selection_metrics")),
                })
    return rows


# ---------------------------------------------------------------------------
# Consolidation driver
# ---------------------------------------------------------------------------
_CONSOLIDATED_RESULTS = "_consolidated_results.parquet"
_CONSOLIDATED_CROSSK = "_consolidated_crossk.parquet"
_CONSOLIDATED_JBL = "_consolidated_jbl.parquet"
_CONSOLIDATION_META = "_consolidation_meta.json"
_RESULTS_UPDATED_SENTINEL = "_results_updated"


def is_stale(nmodels_dir: str) -> bool:
    """Return True if the consolidated Parquet files need rebuilding.

    A directory is considered stale when:
    - No ``_consolidation_meta.json`` exists (never consolidated), OR
    - A ``_results_updated`` sentinel exists and is newer than the meta file
      (experiments have been written since last consolidation).
    """
    meta_path = os.path.join(nmodels_dir, _CONSOLIDATION_META)
    if not os.path.isfile(meta_path):
        return True
    sentinel_path = os.path.join(nmodels_dir, _RESULTS_UPDATED_SENTINEL)
    if not os.path.isfile(sentinel_path):
        return False
    return os.path.getmtime(sentinel_path) > os.path.getmtime(meta_path)


def consolidate_nmodels_dir(nmodels_dir: str, dry_run: bool = False) -> dict:
    """Consolidate a single nmodels directory.  Returns a stats dict."""
    t0 = time.monotonic()

    print(f"  Scanning {nmodels_dir} ...")

    results_rows = _collect_results(nmodels_dir)
    crossk_rows = _collect_crossk(nmodels_dir)
    jbl_rows = _collect_jbl(nmodels_dir)

    elapsed = time.monotonic() - t0
    stats = {
        "nmodels_dir": nmodels_dir,
        "results_count": len(results_rows),
        "crossk_count": len(crossk_rows),
        "jbl_count": len(jbl_rows),
        "scan_seconds": round(elapsed, 1),
    }

    print(
        f"    results={stats['results_count']:,}  "
        f"crossk={stats['crossk_count']:,}  "
        f"jbl={stats['jbl_count']:,}  "
        f"({stats['scan_seconds']}s)"
    )

    if dry_run:
        print("    [dry-run] skipping write")
        return stats

    if results_rows:
        pd.DataFrame(results_rows).to_parquet(
            os.path.join(nmodels_dir, _CONSOLIDATED_RESULTS), index=False,
        )
    if crossk_rows:
        pd.DataFrame(crossk_rows).to_parquet(
            os.path.join(nmodels_dir, _CONSOLIDATED_CROSSK), index=False,
        )
    if jbl_rows:
        pd.DataFrame(jbl_rows).to_parquet(
            os.path.join(nmodels_dir, _CONSOLIDATED_JBL), index=False,
        )

    meta = {
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
        "results_count": len(results_rows),
        "crossk_count": len(crossk_rows),
        "jbl_count": len(jbl_rows),
    }
    with open(os.path.join(nmodels_dir, _CONSOLIDATION_META), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    Wrote Parquet + meta ({time.monotonic() - t0:.1f}s total)")
    return stats


def discover_nmodels_dirs(results_root: str) -> list[str]:
    """Return all nmodels_* directories under *results_root*."""
    dirs = []
    if not os.path.isdir(results_root):
        return dirs
    for sm in sorted(os.listdir(results_root)):
        sm_path = os.path.join(results_root, sm)
        if not os.path.isdir(sm_path):
            continue
        for cs_dir in sorted(os.listdir(sm_path)):
            cs_path = os.path.join(sm_path, cs_dir)
            if not os.path.isdir(cs_path) or not cs_dir.startswith("coreset_"):
                continue
            for nm_dir in sorted(os.listdir(cs_path)):
                nm_path = os.path.join(cs_path, nm_dir)
                if not os.path.isdir(nm_path) or not nm_dir.startswith("nmodels_"):
                    continue
                dirs.append(nm_path)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate benchmark-prediction results into Parquet files.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Root of the results directory tree (default: ./results)",
    )
    parser.add_argument(
        "--nmodels-dir",
        default=None,
        help="Consolidate a single nmodels_* directory instead of the whole tree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing any files",
    )
    parser.add_argument(
        "--force-reconsolidate",
        action="store_true",
        help=(
            "Reconsolidate all directories even if their Parquet files are "
            "already up to date.  By default, a directory is skipped when its "
            "_consolidation_meta.json is newer than the _results_updated "
            "sentinel (written by method_runner.py)."
        ),
    )
    args = parser.parse_args()

    if args.nmodels_dir:
        if not os.path.isdir(args.nmodels_dir):
            print(f"Error: {args.nmodels_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        dirs = [args.nmodels_dir]
    else:
        dirs = discover_nmodels_dirs(args.results_dir)

    if not dirs:
        print(f"No nmodels_* directories found under {args.results_dir}")
        sys.exit(0)

    print(f"Found {len(dirs)} nmodels directories to consolidate")
    if args.dry_run:
        print("[DRY RUN — no files will be written]")
    if not args.force_reconsolidate:
        print("[Skipping up-to-date directories; use --force-reconsolidate to override]")
    print()

    total_t0 = time.monotonic()
    all_stats = []
    skipped = 0
    for d in dirs:
        if not args.force_reconsolidate and not is_stale(d):
            skipped += 1
            continue
        all_stats.append(consolidate_nmodels_dir(d, dry_run=args.dry_run))

    total_elapsed = time.monotonic() - total_t0
    total_results = sum(s["results_count"] for s in all_stats)
    total_crossk = sum(s["crossk_count"] for s in all_stats)
    total_jbl = sum(s["jbl_count"] for s in all_stats)
    parts = [
        f"{len(all_stats)} processed",
    ]
    if skipped:
        parts.append(f"{skipped} skipped (up-to-date)")
    print(
        f"\nDone. {len(dirs)} directories ({', '.join(parts)}), "
        f"{total_results:,} results + {total_crossk:,} crossk + {total_jbl:,} jbl rows "
        f"in {total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
