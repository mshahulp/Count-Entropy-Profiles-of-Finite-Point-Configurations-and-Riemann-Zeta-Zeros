"""
Validated exact count-entropy computation for prepared Riemann-zeta zero blocks.

This stage computes H(L) for:
  - 16 principal zeta blocks: low, 1e12, 1e21, 1e22 (4 blocks each)
  - 4 intermediate-low robustness blocks around zero number 50,000

Window grid:
    L = 0.1, 0.2, ..., 20.0

Primary computation:
    Exact breakpoint integration over the translation-origin interval [0, W-L].
    Breakpoints are {0, W-L} union {x_i} union {x_i-L}, clipped to the domain.
    On every open interval between consecutive breakpoints, N(t;L) is constant.
    The midpoint count is evaluated by np.searchsorted and interval lengths are
    accumulated exactly (up to floating-point arithmetic).

Independent cross-check:
    At selected L values, an event-sweep implementation integrates the active
    origin intervals (x_i-L, x_i] directly. This is mathematically equivalent but
    algorithmically independent from the midpoint-breakpoint implementation.

QC conditions checked for every profile and L:
    sum_k p_L(k) = 1
    p_L(k) >= 0 (within numerical tolerance)
    0 <= H(L) <= log(n+1)
    mean count from pmf is finite
    variance from pmf is nonnegative (within numerical tolerance)

Cross-check L values:
    0.1, 1.0, 5.0, 10.0, 20.0

Outputs:
    zeta_entropy_results/
      zeta_count_entropy_profiles.csv
      zeta_count_entropy_qc.csv
      zeta_count_pmf_sparse.csv
      zeta_profile_summary_by_regime.csv
      zeta_entropy_manifest.json

No GUE comparison is performed in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np


PRINCIPAL_REGIMES = ("low", "1e12", "1e21", "1e22")
ROBUSTNESS_REGIME = "midlow_45k_55k"


def load_block_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    # Works for the generated two-column files.
    x = np.asarray(data["normalized_unfolded_coordinate"], dtype=np.float64)
    if x.ndim != 1:
        x = np.ravel(x)
    if x.size == 0:
        raise ValueError(f"{path}: empty block.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: non-finite coordinates.")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{path}: coordinates not strictly increasing.")
    return x


def exact_pmf_breakpoints(x: np.ndarray, L: float, W: float, n: int) -> np.ndarray:
    """
    Exact sliding-window count pmf via breakpoint partition.

    For t in [0, W-L], N(t;L)=# {x_i in [t,t+L)}.
    Breakpoints arise at t=x_i and t=x_i-L, together with the domain endpoints.
    Between consecutive breakpoints the count is constant.
    """
    T = W - L
    if not (T > 0):
        raise ValueError("Need 0 < L < W.")

    bp = np.concatenate((
        np.array([0.0, T], dtype=np.float64),
        x[(x > 0.0) & (x < T)],
        (x - L)[((x - L) > 0.0) & ((x - L) < T)]
    ))
    bp = np.unique(bp)
    bp.sort()

    lengths = np.diff(bp)
    if np.any(lengths < 0):
        raise RuntimeError("Negative breakpoint interval length.")

    # Zero-length intervals can only occur from roundoff/duplicates already removed.
    keep = lengths > 0.0
    left = bp[:-1][keep]
    right = bp[1:][keep]
    lengths = lengths[keep]
    mid = 0.5 * (left + right)

    # Count x in [mid, mid+L)
    left_idx = np.searchsorted(x, mid, side="left")
    right_idx = np.searchsorted(x, mid + L, side="left")
    counts = right_idx - left_idx

    if counts.min(initial=0) < 0 or counts.max(initial=0) > n:
        raise RuntimeError("Count outside [0,n].")

    measure = np.bincount(counts, weights=lengths, minlength=n + 1).astype(np.float64)
    pmf = measure / T
    return pmf


def exact_pmf_event_sweep(x: np.ndarray, L: float, W: float, n: int) -> np.ndarray:
    """
    Independent exact integration by sweeping active origin intervals.

    A point x_i contributes for t in (x_i-L, x_i], intersected with [0,W-L].
    Endpoints have measure zero, so they do not affect the pmf.
    """
    T = W - L
    if not (T > 0):
        raise ValueError("Need 0 < L < W.")

    events: dict[float, int] = {}
    current = 0

    for xi in x:
        start = max(0.0, float(xi - L))
        end = min(T, float(xi))
        if end <= start:
            continue

        if start <= 0.0:
            current += 1
        else:
            events[start] = events.get(start, 0) + 1

        if end < T:
            events[end] = events.get(end, 0) - 1

    measure = np.zeros(n + 1, dtype=np.float64)
    prev = 0.0

    for pos in sorted(events):
        if pos > prev:
            if current < 0 or current > n:
                raise RuntimeError("Sweep count outside [0,n].")
            measure[current] += pos - prev
        current += events[pos]
        prev = pos

    if prev < T:
        if current < 0 or current > n:
            raise RuntimeError("Sweep count outside [0,n] at tail.")
        measure[current] += T - prev

    return measure / T


def entropy_from_pmf(pmf: np.ndarray) -> float:
    positive = pmf > 0.0
    p = pmf[positive]
    return float(-np.sum(p * np.log(p)))


def pmf_moments(pmf: np.ndarray) -> tuple[float, float]:
    k = np.arange(pmf.size, dtype=np.float64)
    mean = float(np.dot(k, pmf))
    var = float(np.dot((k - mean) ** 2, pmf))
    return mean, var


def find_block_files(principal_dir: Path, robustness_dir: Path) -> list[tuple[str, int, str, Path]]:
    """
    Returns tuples:
        (regime, block_number, role, path)
    role is principal or robustness.
    """
    found = []

    for regime in PRINCIPAL_REGIMES:
        for b in range(1, 5):
            path = principal_dir / f"{regime}_block{b:02d}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing principal block: {path}")
            found.append((regime, b, "principal", path))

    for b in range(1, 5):
        path = robustness_dir / f"midlow_block{b:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing robustness block: {path}")
        found.append((ROBUSTNESS_REGIME, b, "robustness", path))

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--principal-block-dir",
        default="zeta_prepared/blocks",
        help="Directory containing the 16 principal normalized block CSV files."
    )
    parser.add_argument(
        "--robustness-block-dir",
        default="zeta_lowheight_robustness/blocks",
        help="Directory containing the 4 intermediate-low robustness block CSV files."
    )
    parser.add_argument("--output-dir", default="zeta_entropy_results")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--W", type=float, default=2000.0)
    parser.add_argument("--L-min", type=float, default=0.1)
    parser.add_argument("--L-max", type=float, default=20.0)
    parser.add_argument("--L-step", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=5e-12)
    args = parser.parse_args()

    principal_dir = Path(args.principal_block_dir)
    robustness_dir = Path(args.robustness_block_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = args.n
    W = args.W
    tol = args.tol

    # Integer construction avoids cumulative arange drift.
    nL = int(round((args.L_max - args.L_min) / args.L_step)) + 1
    L_grid = np.array(
        [args.L_min + i * args.L_step for i in range(nL)],
        dtype=np.float64
    )
    L_grid = np.round(L_grid, 12)

    crosscheck_targets = np.array([0.1, 1.0, 5.0, 10.0, 20.0], dtype=np.float64)

    block_files = find_block_files(principal_dir, robustness_dir)

    profile_rows = []
    qc_rows = []
    pmf_rows = []

    print(f"Found {len(block_files)} zeta blocks.")
    print(f"Computing {len(L_grid)} L values per block: {L_grid[0]} ... {L_grid[-1]}")
    print("")

    for block_idx, (regime, block_no, role, path) in enumerate(block_files, start=1):
        x = load_block_csv(path)

        if x.size != n:
            raise ValueError(f"{path}: expected n={n}, found {x.size}.")
        if x[0] <= 0 or x[-1] >= W:
            raise ValueError(
                f"{path}: retained coordinates must lie strictly inside (0,W). "
                f"Observed min={x[0]}, max={x[-1]}."
            )

        print(f"[{block_idx:02d}/{len(block_files)}] {regime} block {block_no}")

        for L in L_grid:
            pmf = exact_pmf_breakpoints(x, float(L), W, n)

            pmf_sum = float(np.sum(pmf))
            pmf_min = float(np.min(pmf))
            pmf_max = float(np.max(pmf))
            H = entropy_from_pmf(pmf)
            mean_count, variance = pmf_moments(pmf)

            support = np.flatnonzero(pmf > 0)
            support_min = int(support[0]) if support.size else -1
            support_max = int(support[-1]) if support.size else -1
            support_size = int(support.size)

            # Core QC
            sum_error = abs(pmf_sum - 1.0)
            nonnegative_ok = pmf_min >= -tol
            sum_ok = sum_error <= tol
            entropy_ok = (-tol <= H <= math.log(n + 1) + tol)
            variance_ok = variance >= -tol
            finite_ok = all(map(math.isfinite, [pmf_sum, H, mean_count, variance]))

            # Independent exact cross-check at selected L.
            is_crosscheck = bool(np.any(np.isclose(L, crosscheck_targets, atol=1e-12, rtol=0)))
            crosscheck_tv = ""
            crosscheck_max_abs = ""
            crosscheck_ok = ""

            if is_crosscheck:
                pmf2 = exact_pmf_event_sweep(x, float(L), W, n)
                tv = 0.5 * float(np.sum(np.abs(pmf - pmf2)))
                maxabs = float(np.max(np.abs(pmf - pmf2)))
                crosscheck_tv = tv
                crosscheck_max_abs = maxabs
                crosscheck_ok = (tv <= 5 * tol)

            all_ok = (
                nonnegative_ok and sum_ok and entropy_ok and variance_ok and finite_ok
                and (crosscheck_ok is not False)
            )

            profile_rows.append({
                "role": role,
                "regime": regime,
                "block": block_no,
                "L": float(L),
                "H": H,
                "mean_count": mean_count,
                "number_variance": max(0.0, variance),
                "support_min_k": support_min,
                "support_max_k": support_max,
                "support_size": support_size,
            })

            qc_rows.append({
                "role": role,
                "regime": regime,
                "block": block_no,
                "L": float(L),
                "pmf_sum": pmf_sum,
                "pmf_sum_abs_error": sum_error,
                "pmf_min": pmf_min,
                "pmf_max": pmf_max,
                "entropy": H,
                "entropy_upper_bound_log_nplus1": math.log(n + 1),
                "mean_count": mean_count,
                "number_variance": variance,
                "support_size": support_size,
                "crosscheck_performed": is_crosscheck,
                "crosscheck_tv": crosscheck_tv,
                "crosscheck_max_abs": crosscheck_max_abs,
                "crosscheck_ok": crosscheck_ok,
                "all_qc_ok": all_ok,
            })

            # Sparse pmf archive: only positive masses.
            positive_idx = np.flatnonzero(pmf > 0)
            for k in positive_idx:
                pmf_rows.append({
                    "role": role,
                    "regime": regime,
                    "block": block_no,
                    "L": float(L),
                    "k": int(k),
                    "p": float(pmf[k]),
                })

    # Fail loudly if any QC condition did not pass.
    failed = [r for r in qc_rows if not r["all_qc_ok"]]
    if failed:
        fail_path = outdir / "zeta_count_entropy_qc_FAILED.csv"
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qc_rows)
        raise RuntimeError(
            f"{len(failed)} QC rows failed. Full QC written to {fail_path}. "
            "Do not use the entropy results until investigated."
        )

    # Write per-block profiles.
    profile_path = outdir / "zeta_count_entropy_profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    # Write QC.
    qc_path = outdir / "zeta_count_entropy_qc.csv"
    with qc_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qc_rows)

    # Write sparse PMFs.
    pmf_path = outdir / "zeta_count_pmf_sparse.csv"
    with pmf_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pmf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pmf_rows)

    # Regime-level summaries at each L.
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in profile_rows:
        key = (row["role"], row["regime"], row["L"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (role, regime, L), rows in sorted(grouped.items(), key=lambda z: (z[0][0], z[0][1], z[0][2])):
        Hvals = np.array([r["H"] for r in rows], dtype=float)
        Vvals = np.array([r["number_variance"] for r in rows], dtype=float)
        summary_rows.append({
            "role": role,
            "regime": regime,
            "L": L,
            "num_blocks": len(rows),
            "H_mean": float(np.mean(Hvals)),
            "H_sd_across_blocks": float(np.std(Hvals, ddof=1)) if len(rows) > 1 else 0.0,
            "H_min": float(np.min(Hvals)),
            "H_max": float(np.max(Hvals)),
            "number_variance_mean": float(np.mean(Vvals)),
            "number_variance_sd_across_blocks": float(np.std(Vvals, ddof=1)) if len(rows) > 1 else 0.0,
        })

    summary_path = outdir / "zeta_profile_summary_by_regime.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Compact global QC summary for manifest.
    max_pmf_sum_error = max(r["pmf_sum_abs_error"] for r in qc_rows)
    min_pmf_value = min(r["pmf_min"] for r in qc_rows)

    cross_tvs = [
        float(r["crosscheck_tv"])
        for r in qc_rows
        if r["crosscheck_performed"]
    ]
    cross_maxabs = [
        float(r["crosscheck_max_abs"])
        for r in qc_rows
        if r["crosscheck_performed"]
    ]

    manifest = {
        "purpose": "validated exact zeta count-entropy computation",
        "principal_block_dir": str(principal_dir.as_posix()),
        "robustness_block_dir": str(robustness_dir.as_posix()),
        "n": n,
        "W": W,
        "L_min": float(L_grid[0]),
        "L_max": float(L_grid[-1]),
        "L_step": args.L_step,
        "num_L_values": int(len(L_grid)),
        "num_principal_blocks": 16,
        "num_robustness_blocks": 4,
        "num_total_blocks": 20,
        "num_entropy_evaluations": int(20 * len(L_grid)),
        "primary_algorithm": "exact breakpoint partition with midpoint counts via numpy.searchsorted",
        "independent_crosscheck_algorithm": "event sweep of active origin intervals (x_i-L, x_i]",
        "crosscheck_L_values": crosscheck_targets.tolist(),
        "qc_tolerance": tol,
        "all_qc_passed": True,
        "max_pmf_sum_abs_error": max_pmf_sum_error,
        "minimum_pmf_value": min_pmf_value,
        "max_crosscheck_total_variation": max(cross_tvs) if cross_tvs else None,
        "max_crosscheck_pointwise_abs_difference": max(cross_maxabs) if cross_maxabs else None,
        "entropy_log_base": "natural",
        "number_variance_computed_from_same_pmf": True,
        "gue_comparison_performed_here": False,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "outputs": {
            "profiles": str(profile_path.as_posix()),
            "qc": str(qc_path.as_posix()),
            "sparse_pmf": str(pmf_path.as_posix()),
            "regime_summary": str(summary_path.as_posix()),
        },
    }

    manifest_path = outdir / "zeta_entropy_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("SUCCESS: all entropy computations and QC checks passed.")
    print(f"Profiles: {profile_path.resolve()}")
    print(f"QC:       {qc_path.resolve()}")
    print(f"PMFs:     {pmf_path.resolve()}")
    print(f"Summary:  {summary_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print("")
    print(f"Maximum |sum(p)-1|: {max_pmf_sum_error:.3e}")
    print(f"Minimum pmf value:   {min_pmf_value:.3e}")
    print(f"Max cross-check TV:  {max(cross_tvs):.3e}")
    print("")
    print("STOP HERE before GUE comparison. Review QC and zeta profiles first.")


if __name__ == "__main__":
    main()
