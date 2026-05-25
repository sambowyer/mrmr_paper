#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Precompute MI cache (relevance vectors) for all discrete
# (openllm-leaderboard) datasets.  --only-discrete is on by default so
# continuous pass@k datasets are skipped automatically.
#
# Safe to re-run: already-computed cache files are skipped.

python3 precompute_mi_cache.py --workers 32
