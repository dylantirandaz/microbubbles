from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _compound(path: Path, acq: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5[f"acquisitions/{acq}/meta/compound_image"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mach", required=True, type=Path)
    parser.add_argument("--mlx", required=True, type=Path)
    parser.add_argument("--acq", type=int, default=0)
    args = parser.parse_args()

    mach = _compound(args.mach, args.acq)
    mlx = _compound(args.mlx, args.acq)
    if mach.shape != mlx.shape:
        raise ValueError(f"shape mismatch: MACH {mach.shape}, MLX {mlx.shape}")

    delta = mach - mlx
    mach_norm = float(np.linalg.norm(mach))
    rel_l2 = float(np.linalg.norm(delta) / mach_norm) if mach_norm else float("nan")
    mag_corr = float(np.corrcoef(np.abs(mach).ravel(), np.abs(mlx).ravel())[0, 1])

    print(f"shape: {mach.shape}")
    print(f"max_abs: {float(np.max(np.abs(delta))):.6g}")
    print(f"mean_abs: {float(np.mean(np.abs(delta))):.6g}")
    print(f"relative_l2: {rel_l2:.6g}")
    print(f"magnitude_corr: {mag_corr:.6g}")


if __name__ == "__main__":
    main()
