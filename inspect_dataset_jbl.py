#!/usr/bin/env python3
"""Quick inspection of dataset score .jbl files."""

import sys
import os
import joblib as jbl
import numpy as np

def inspect(path: str):
    data = jbl.load(path)
    print(f"\n{'=' * 60}")
    print(f"File: {path}")
    print(f"Type: {type(data).__name__}")

    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {np.nanmin(data):.4f}  Max: {np.nanmax(data):.4f}  Mean: {np.nanmean(data):.4f}")
        print(f"NaNs: {np.isnan(data).sum()}" if np.issubdtype(data.dtype, np.floating) else "")
        print(f"\nFirst 10 rows (truncated to 10 cols):\n{data[:10, :10]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"Contents: {data}")
    else:
        print(repr(data))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.jbl> [file2.jbl ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        inspect(path)
