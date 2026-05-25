#!/usr/bin/env python3
"""Precompute notebook summary tables for faster plotting.

Builds standard summary DataFrames (all methods) for:
- Binary datasets on split_method="binned_interpolation"
- Continuous datasets on split_method="stratified"
- Cross-k (pass@k) summaries on split_method="stratified" for source_mode in {"k1", "opt"}

for coreset sizes {5%, 10%, 15%} and num_train_models:
- {15, 30, 50} for binary
- {16, 32, 52} for continuous

Outputs are written to mrmr/viz/notebooks/cache/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import nb_utils


TARGET_CORESET_SIZES = {"5%", "10%", "15%"}
TARGET_NMODELS_BINARY = {"15", "30", "50"}
TARGET_NMODELS_CONTINUOUS = {"16", "32", "52"}
TARGET_NMODELS_PASSK = {"10", "15", "20"}


def _target_settings(split_method: str, allowed_nmodels: set[str]) -> list[nb_utils.Setting]:
    settings = []
    for setting in nb_utils.discover_settings(split_method=split_method):
        if setting.coreset_size not in TARGET_CORESET_SIZES:
            continue
        if setting.num_train_models not in allowed_nmodels:
            continue
        settings.append(setting)
    settings.sort(key=nb_utils._setting_sort_key)
    return settings


def _build_standard_summary_for_group(
    *,
    group_name: str,
    split_method: str,
    datasets: list[str],
    allowed_nmodels: set[str],
) -> pd.DataFrame:
    settings = _target_settings(split_method, allowed_nmodels)
    if not settings:
        tqdm.write(
            f"[warn] No settings found for group={group_name} split={split_method} "
            f"coresets={sorted(TARGET_CORESET_SIZES)} nmodels={sorted(allowed_nmodels)}"
        )
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for setting in tqdm(settings, desc=f"{group_name}: summarize settings", unit="setting"):
        summary = nb_utils.summarize_setting_standard(
            setting=setting,
            datasets=datasets,
            methods=None,  # all methods
        )
        if summary.empty:
            continue
        frame = summary.copy()
        frame.insert(0, "group", group_name)
        frame.insert(1, "split_method", split_method)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_passk_summary_for_group(
    *,
    group_name: str,
    split_method: str,
    benchmarks: list[str],
    allowed_nmodels: set[str],
    source_mode: str,
) -> pd.DataFrame:
    settings = _target_settings(split_method, allowed_nmodels)
    if not settings:
        tqdm.write(
            f"[warn] No settings found for group={group_name} split={split_method} "
            f"coresets={sorted(TARGET_CORESET_SIZES)} nmodels={sorted(allowed_nmodels)}"
        )
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for setting in tqdm(
        settings,
        desc=f"{group_name}: summarize passk ({source_mode})",
        unit="setting",
    ):
        summary = nb_utils.summarize_setting_passk(
            setting=setting,
            benchmarks=benchmarks,
            methods=None,  # all methods
            source_mode=source_mode,
        )
        if summary.empty:
            continue
        frame = summary.copy()
        frame.insert(0, "group", group_name)
        frame.insert(1, "split_method", split_method)
        frame.insert(2, "source_mode", source_mode)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_all_summaries(output_dir: Path) -> dict[str, Path]:
    nb_utils.ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_datasets = list(nb_utils.BINARY_DATASETS_DEFAULT)
    continuous_datasets = list(nb_utils.CONTINUOUS_DATASETS_DEFAULT)
    passk_benchmarks = list(nb_utils.PASSK_BENCHMARKS_DEFAULT)

    binary_summary = _build_standard_summary_for_group(
        group_name="binary",
        split_method=nb_utils.BINARY_SPLIT_METHOD,
        datasets=binary_datasets,
        allowed_nmodels=TARGET_NMODELS_BINARY,
    )
    continuous_summary = _build_standard_summary_for_group(
        group_name="continuous",
        split_method=nb_utils.CONTINUOUS_SPLIT_METHOD,
        datasets=continuous_datasets,
        allowed_nmodels=TARGET_NMODELS_CONTINUOUS,
    )
    passk_k1_summary = _build_passk_summary_for_group(
        group_name="passk",
        split_method=nb_utils.CONTINUOUS_SPLIT_METHOD,
        benchmarks=passk_benchmarks,
        allowed_nmodels=TARGET_NMODELS_PASSK,
        source_mode="k1",
    )
    passk_opt_summary = _build_passk_summary_for_group(
        group_name="passk",
        split_method=nb_utils.CONTINUOUS_SPLIT_METHOD,
        benchmarks=passk_benchmarks,
        allowed_nmodels=TARGET_NMODELS_PASSK,
        source_mode="opt",
    )

    combined_frames = [df for df in [binary_summary, continuous_summary] if not df.empty]
    combined_summary = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    passk_combined_frames = [df for df in [passk_k1_summary, passk_opt_summary] if not df.empty]
    passk_combined_summary = (
        pd.concat(passk_combined_frames, ignore_index=True) if passk_combined_frames else pd.DataFrame()
    )

    out_paths = {
        "binary": output_dir / "summary_standard_binary.parquet",
        "continuous": output_dir / "summary_standard_continuous.parquet",
        "combined": output_dir / "summary_standard_global.parquet",
        "passk_k1": output_dir / "summary_passk_k1.parquet",
        "passk_opt": output_dir / "summary_passk_opt.parquet",
        "passk_combined": output_dir / "summary_passk_global.parquet",
    }

    binary_summary.to_parquet(out_paths["binary"], index=False)
    continuous_summary.to_parquet(out_paths["continuous"], index=False)
    combined_summary.to_parquet(out_paths["combined"], index=False)
    passk_k1_summary.to_parquet(out_paths["passk_k1"], index=False)
    passk_opt_summary.to_parquet(out_paths["passk_opt"], index=False)
    passk_combined_summary.to_parquet(out_paths["passk_combined"], index=False)
    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute notebook standard and cross-k summaries for selected split/settings."
    )
    parser.add_argument(
        "--output-dir",
        default=str(nb_utils.CACHE_DIR),
        help="Directory to write parquet summary outputs (default: notebooks cache dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    out_paths = build_all_summaries(output_dir=output_dir)
    print("Wrote summary files:")
    for name, path in out_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
