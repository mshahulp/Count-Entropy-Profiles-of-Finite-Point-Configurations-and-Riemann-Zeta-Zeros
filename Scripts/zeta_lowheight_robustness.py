"""
Intermediate-low-height robustness preparation for the count-entropy study.

Purpose
-------
Construct an additional 10,000-zero parent segment from the already-downloaded
Odlyzko `zeros1` file, centred approximately on zero number 50,000, and process it
with exactly the same unfolding, guard-point, boundary, and normalization rules
used for the principal low-height dataset.

This is a ROBUSTNESS dataset only. It does not replace the principal first-10,000
low-height dataset.

Parent segment
--------------
Global zero numbers 45,001 ... 55,000 (10,000 consecutive zeros).

Four non-overlapping retained blocks of n=2000 are then formed using the same
0-based retained starts within the parent segment:

    1, 2001, 4001, 6001

which correspond to retained global zero-number ranges:

    45,002 ... 47,001
    47,002 ... 49,001
    49,002 ... 51,001
    51,002 ... 53,001

Each block has one immediate guard zero on either side.

The script deliberately does NOT compute count entropy.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
import platform
from pathlib import Path

import mpmath as mp
import numpy as np


NUMERIC_LINE_CHARS = set("0123456789+-.eE")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_full_ordinate_table(path: Path) -> list[mp.mpf]:
    vals = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # zeros1 is one full ordinate per numeric line.
            if all(ch in NUMERIC_LINE_CHARS for ch in s):
                try:
                    vals.append(mp.mpf(s))
                except ValueError:
                    pass
    return vals


def nbar(T: mp.mpf) -> mp.mpf:
    two_pi = 2 * mp.pi
    return (T / two_pi) * mp.log(T / two_pi) - (T / two_pi) + mp.mpf(7) / 8


def make_block(parent: list[mp.mpf], start: int, n: int) -> dict:
    local_gamma = parent[start - 1 : start + n + 1]
    if len(local_gamma) != n + 2:
        raise RuntimeError("Unexpected block length; missing guard point.")

    gamma_gaps = [local_gamma[i + 1] - local_gamma[i]
                  for i in range(len(local_gamma) - 1)]
    if not all(g > 0 for g in gamma_gaps):
        raise RuntimeError("Source ordinates are not strictly increasing.")

    origin = nbar(local_gamma[0])
    unfolded = [nbar(g) - origin for g in local_gamma]

    left_guard = unfolded[0]
    retained = unfolded[1:-1]
    right_guard = unfolded[-1]

    a = (left_guard + retained[0]) / 2
    b = (retained[-1] + right_guard) / 2
    W_pre = b - a
    scale = mp.mpf(n) / W_pre

    normalized_mp = [(x - a) * scale for x in retained]
    normalized64 = np.array([float(x) for x in normalized_mp], dtype=np.float64)

    if not np.all(np.diff(normalized64) > 0):
        raise RuntimeError("Monotonicity lost after float64 conversion.")

    max_float64_error = max(abs(mp.mpf(float(y)) - y) for y in normalized_mp)

    return {
        "normalized64": normalized64,
        "normalized_mp": normalized_mp,
        "gamma_gaps": gamma_gaps,
        "W_pre": W_pre,
        "scale": scale,
        "max_float64_error": max_float64_error,
        "first_gamma": local_gamma[1],
        "last_gamma": local_gamma[-2],
    }


def save_block(path: Path, arr: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "normalized_unfolded_coordinate"])
        for i, x in enumerate(arr, 1):
            w.writerow([i, format(float(x), ".17g")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-low-file",
        default="zeta_prepared/raw/low.txt",
        help="Path to the already-downloaded Odlyzko zeros1 file."
    )
    parser.add_argument(
        "--output-dir",
        default="zeta_lowheight_robustness"
    )
    parser.add_argument("--mp-dps", type=int, default=80)
    args = parser.parse_args()

    mp.mp.dps = args.mp_dps

    raw_path = Path(args.raw_low_file)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Cannot find {raw_path}. Run zeta_prepare_validate.py first, "
            "or supply --raw-low-file with the correct path."
        )

    outdir = Path(args.output_dir)
    blockdir = outdir / "blocks"
    blockdir.mkdir(parents=True, exist_ok=True)

    all_zeros = parse_full_ordinate_table(raw_path)
    if len(all_zeros) < 55_000:
        raise RuntimeError(
            f"Parsed only {len(all_zeros)} zeros; at least 55,000 are required."
        )

    # Python slice [45000:55000] corresponds to global zero numbers 45,001..55,000.
    parent_start_global = 45_001
    parent_end_global = 55_000
    parent = all_zeros[45_000:55_000]

    if len(parent) != 10_000:
        raise RuntimeError("Intermediate parent segment does not contain 10,000 zeros.")

    n = 2000
    block_starts = (1, 2001, 4001, 6001)

    rows = []
    for block_no, start in enumerate(block_starts, 1):
        result = make_block(parent, start, n)

        global_start = parent_start_global + start
        global_end = global_start + n - 1

        outpath = blockdir / f"midlow_block{block_no:02d}.csv"
        save_block(outpath, result["normalized64"])

        final_gaps = np.diff(result["normalized64"])
        raw_retained_gaps = result["gamma_gaps"][1:-1]
        raw_retained_gaps64 = np.array([float(g) for g in raw_retained_gaps], dtype=float)

        rows.append({
            "dataset": "midlow_45k_55k",
            "block": block_no,
            "parent_global_start": parent_start_global,
            "parent_global_end": parent_end_global,
            "global_zero_start": global_start,
            "global_zero_end": global_end,
            "retained_n": n,
            "source_monotone": True,
            "first_gamma": mp.nstr(result["first_gamma"], 40),
            "last_gamma": mp.nstr(result["last_gamma"], 40),
            "raw_mean_gap_gamma": float(np.mean(raw_retained_gaps64)),
            "raw_min_gap_gamma": float(np.min(raw_retained_gaps64)),
            "raw_max_gap_gamma": float(np.max(raw_retained_gaps64)),
            "pre_affine_W": float(result["W_pre"]),
            "affine_scale": float(result["scale"]),
            "final_W": float(n),
            "final_intensity_n_over_W": 1.0,
            "final_mean_nn_gap": float(np.mean(final_gaps)),
            "final_min_nn_gap": float(np.min(final_gaps)),
            "final_max_nn_gap": float(np.max(final_gaps)),
            "final_coordinate_min": float(result["normalized64"][0]),
            "final_coordinate_max": float(result["normalized64"][-1]),
            "float64_monotone_after_normalization": bool(np.all(final_gaps > 0)),
            "max_abs_error_from_final_float64_conversion": float(result["max_float64_error"]),
            "mp_dps": args.mp_dps,
            "block_file": str(outpath.as_posix()),
        })

    validation_path = outdir / "midlow_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "purpose": "low-height robustness only; does not replace principal first-10000 dataset",
        "source_raw_file": str(raw_path.as_posix()),
        "source_raw_sha256": sha256_file(raw_path),
        "source_global_table": "Odlyzko zeros1 (first 100,000 zeros)",
        "parent_global_zero_start": parent_start_global,
        "parent_global_zero_end": parent_end_global,
        "parent_count": 10_000,
        "n": n,
        "block_starts_within_parent_0_based": list(block_starts),
        "mp_dps": args.mp_dps,
        "unfolding": "Nbar(T)=T/(2*pi)*log(T/(2*pi))-T/(2*pi)+7/8",
        "oscillatory_S_T_included": False,
        "boundary_rule": "midpoints to one immediate guard zero on each side",
        "final_interval": "[0,2000]",
        "entropy_computed_here": False,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
    }

    manifest_path = outdir / "midlow_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Intermediate-low-height robustness preparation complete.")
    print(f"Parent zeros: {parent_start_global:,} ... {parent_end_global:,}")
    print(f"Validation: {validation_path.resolve()}")
    print(f"Manifest:   {manifest_path.resolve()}")
    print(f"Blocks:     {blockdir.resolve()}")
    print("\nSTOP HERE. Do not compute entropy yet.")


if __name__ == "__main__":
    main()
