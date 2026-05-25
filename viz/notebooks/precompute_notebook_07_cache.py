#!/usr/bin/env python3
"""Precompute cache artifacts used by notebook 07 winner maps."""

from __future__ import annotations

import argparse

import nb_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute notebook 07 winner-map cache files in mrmr/viz/notebooks/cache. "
            "Defaults to binary datasets only."
        )
    )
    parser.add_argument(
        "--include-continuous",
        action="store_true",
        help="Also precompute continuous-dataset winner-map cache.",
    )
    parser.add_argument(
        "--include-passk",
        action="store_true",
        help="Also precompute pass@k winner-map cache (k1 and opt source modes).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing cache files and rebuild from raw results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_paths = nb_utils.precompute_notebook_07_cache(
        include_continuous=args.include_continuous,
        include_passk=args.include_passk,
        force_recompute=args.force_recompute,
    )
    print("Notebook 07 cache files:")
    for name, path in cache_paths.items():
        status = "exists" if path.is_file() else "missing"
        print(f"  - {name}: {path} [{status}]")


if __name__ == "__main__":
    main()
