"""
Gap-shuffled surrogate experiment for the count-entropy manuscript.

Scientific question
-------------------
Can the sliding-window count-entropy profile distinguish a point configuration
from surrogates having exactly the same empirical nearest-neighbour gap multiset
but randomized gap order?

For each prepared zeta block X={x_1<...<x_n}, define gaps
    g_i = x_{i+1}-x_i.
A surrogate is reconstructed from a random permutation pi:
    x*_1 = x_1,
    x*_{j+1} = x*_j + g_{pi(j)}.

Thus each surrogate preserves:
  - n,
  - the empirical nearest-neighbour gap multiset exactly by construction,
  - x_1 and (up to summation roundoff) x_n,
  - the same observation interval [0,W],

while randomizing the sequential ordering of the gaps.

This does NOT claim to preserve every one-point or two-point statistic, nor does
it prove that every higher-order dependency is destroyed. It is specifically a
gap-order randomization experiment.

Primary outputs include BOTH:
  1. Shannon count entropy H(L), and
  2. number variance Sigma^2(L),

because both are functionals of the same sliding-window count pmf. This lets the
surrogate experiment assess whether either/both statistics respond to gap order.

Defaults
--------
n = 2000
W = 2000
L = 0.1, 0.2, ..., 20.0
R_surrogate = 200 per zeta block
seed = 20260901

The script includes all 16 principal blocks and the four intermediate-low
robustness blocks by default.

For each block it:
  - recomputes the original exact count profile;
  - verifies it against zeta_count_entropy_profiles.csv;
  - generates 200 independent gap permutations;
  - computes exact H(L) and Sigma^2(L) for every surrogate;
  - constructs pointwise 95% surrogate bands;
  - computes original-to-surrogate-mean RMSE and sup distances;
  - builds leave-one-out surrogate self-distance distributions;
  - reports descriptive empirical upper-tail fractions.

The upper-tail fractions are reference-tail diagnostics, not formal hypothesis
test p-values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PRINCIPAL_REGIMES = ("low", "1e12", "1e21", "1e22")
ROBUSTNESS_REGIME = "midlow_45k_55k"


def load_block_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if "normalized_unfolded_coordinate" not in data.dtype.names:
        raise ValueError(
            f"{path}: expected column 'normalized_unfolded_coordinate'; "
            f"found {data.dtype.names}"
        )
    x = np.asarray(data["normalized_unfolded_coordinate"], dtype=np.float64)
    x = np.ravel(x)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: empty or non-finite block.")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{path}: coordinates are not strictly increasing.")
    return x


def find_block_files(principal_dir: Path, robustness_dir: Path):
    found = []
    for regime in PRINCIPAL_REGIMES:
        for b in range(1, 5):
            p = principal_dir / f"{regime}_block{b:02d}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing principal block: {p}")
            found.append(("principal", regime, b, p))
    for b in range(1, 5):
        p = robustness_dir / f"midlow_block{b:02d}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing robustness block: {p}")
        found.append(("robustness", ROBUSTNESS_REGIME, b, p))
    return found


def exact_entropy_variance(x: np.ndarray, L: float, W: float, n: int):
    """
    Exact breakpoint integration of the translation-induced count law.
    """
    T = W - L
    if not (0.0 < L < W):
        raise ValueError("Need 0 < L < W.")

    shifted = x - L
    bp = np.concatenate((
        np.array([0.0, T], dtype=np.float64),
        x[(x > 0.0) & (x < T)],
        shifted[(shifted > 0.0) & (shifted < T)],
    ))
    bp = np.unique(bp)
    bp.sort()

    lengths = np.diff(bp)
    keep = lengths > 0.0
    left = bp[:-1][keep]
    right = bp[1:][keep]
    lengths = lengths[keep]
    mid = 0.5 * (left + right)

    li = np.searchsorted(x, mid, side="left")
    ri = np.searchsorted(x, mid + L, side="left")
    counts = ri - li

    measure = np.bincount(counts, weights=lengths, minlength=n + 1).astype(float)
    pmf = measure / T

    ps = float(np.sum(pmf))
    if abs(ps - 1.0) > 5e-12 or np.min(pmf) < -5e-12:
        raise RuntimeError(f"Invalid pmf at L={L}: sum={ps}, min={np.min(pmf)}")

    positive = pmf > 0.0
    p = pmf[positive]
    H = float(-np.sum(p * np.log(p)))

    k = np.arange(pmf.size, dtype=float)
    mean = float(np.dot(k, pmf))
    var = float(np.dot((k - mean) ** 2, pmf))
    if var < -5e-12:
        raise RuntimeError(f"Negative variance at L={L}: {var}")

    return H, max(0.0, var)


def compute_profile(x: np.ndarray, L_grid: np.ndarray, W: float, n: int):
    H = np.empty(len(L_grid), dtype=float)
    V = np.empty(len(L_grid), dtype=float)
    for j, L in enumerate(L_grid):
        H[j], V[j] = exact_entropy_variance(x, float(L), W, n)
    return H, V


def build_surrogate(x: np.ndarray, rng: np.random.Generator):
    gaps = np.diff(x)
    perm = rng.permutation(gaps.size)
    permuted_gaps = gaps[perm]

    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[0] + np.cumsum(permuted_gaps, dtype=np.float64)

    if not np.all(np.diff(y) > 0):
        raise RuntimeError("Surrogate lost strict ordering.")

    return y, permuted_gaps, perm


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def supdist(a, b):
    return float(np.max(np.abs(a - b)))


def empirical_upper_tail(ref: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(ref >= observed)) / (len(ref) + 1))


def empirical_percentile(ref: np.ndarray, observed: float) -> float:
    return float(100.0 * np.mean(ref <= observed))


def load_validated_original_profiles(path: Path):
    df = pd.read_csv(path)
    required = {
        "role", "regime", "block", "L", "H", "number_variance"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Original profile CSV missing columns: {sorted(missing)}")
    return df


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--principal-block-dir", required=True)
    parser.add_argument("--robustness-block-dir", required=True)
    parser.add_argument("--original-profiles", required=True,
                        help="Validated zeta_count_entropy_profiles.csv")
    parser.add_argument("--output-dir", default="gap_shuffled_surrogate_results")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--W", type=float, default=2000.0)
    parser.add_argument("--L-min", type=float, default=0.1)
    parser.add_argument("--L-max", type=float, default=20.0)
    parser.add_argument("--L-step", type=float, default=0.1)
    parser.add_argument("--R-surrogate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    principal_dir = Path(args.principal_block_dir)
    robustness_dir = Path(args.robustness_block_dir)
    original_path = Path(args.original_profiles)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = args.n
    W = args.W
    R = args.R_surrogate

    nL = int(round((args.L_max - args.L_min) / args.L_step)) + 1
    L_grid = np.array(
        [args.L_min + i * args.L_step for i in range(nL)], dtype=float
    )
    L_grid = np.round(L_grid, 12)

    original_df = load_validated_original_profiles(original_path)
    blocks = find_block_files(principal_dir, robustness_dir)

    # SeedSequence creates deterministic independent RNG streams by block.
    master_ss = np.random.SeedSequence(args.seed)
    block_seeds = master_ss.spawn(len(blocks))

    block_rows = []
    pointwise_rows = []
    self_rows = []

    # Compact numeric archive.
    all_surrogate_H = np.empty((len(blocks), R, len(L_grid)), dtype=np.float64)
    all_surrogate_V = np.empty((len(blocks), R, len(L_grid)), dtype=np.float64)
    all_original_H = np.empty((len(blocks), len(L_grid)), dtype=np.float64)
    all_original_V = np.empty((len(blocks), len(L_grid)), dtype=np.float64)
    roles, regimes, block_numbers = [], [], []

    global_max_original_H_mismatch = 0.0
    global_max_original_V_mismatch = 0.0
    global_max_endpoint_drift = 0.0
    global_max_gap_multiset_error = 0.0

    print(f"Blocks: {len(blocks)}")
    print(f"Surrogates per block: {R}")
    print(f"L grid: {L_grid[0]} ... {L_grid[-1]} ({len(L_grid)} values)")
    print(f"Total surrogate profiles: {len(blocks) * R}")
    print("")

    for bi, (role, regime, block, path) in enumerate(blocks):
        print(f"[{bi+1:02d}/{len(blocks)}] {role} {regime} block {block}")

        x = load_block_csv(path)
        if len(x) != n:
            raise ValueError(f"{path}: expected n={n}, got {len(x)}")
        if x[0] <= 0.0 or x[-1] >= W:
            raise ValueError(f"{path}: expected retained points strictly inside (0,W).")

        orig_H, orig_V = compute_profile(x, L_grid, W, n)

        # Cross-check original against already validated prior computation.
        old = original_df[
            (original_df["role"] == role)
            & (original_df["regime"] == regime)
            & (original_df["block"] == block)
        ].sort_values("L")

        if len(old) != len(L_grid):
            raise ValueError(
                f"Original-profile row count mismatch for {regime} block {block}."
            )
        old_L = old["L"].to_numpy(float)
        if not np.allclose(old_L, L_grid, atol=1e-12, rtol=0):
            raise ValueError(f"Original L-grid mismatch for {regime} block {block}.")

        H_mismatch = float(np.max(np.abs(orig_H - old["H"].to_numpy(float))))
        V_mismatch = float(np.max(np.abs(orig_V - old["number_variance"].to_numpy(float))))
        global_max_original_H_mismatch = max(global_max_original_H_mismatch, H_mismatch)
        global_max_original_V_mismatch = max(global_max_original_V_mismatch, V_mismatch)

        if H_mismatch > 5e-12 or V_mismatch > 5e-12:
            raise RuntimeError(
                f"Recomputed original profile disagrees with validated archive: "
                f"H={H_mismatch}, V={V_mismatch}"
            )

        rng = np.random.default_rng(block_seeds[bi])
        original_gaps = np.diff(x)
        original_sorted_gaps = np.sort(original_gaps)

        SH = np.empty((R, len(L_grid)), dtype=float)
        SV = np.empty((R, len(L_grid)), dtype=float)

        for r in range(R):
            y, pgaps, perm = build_surrogate(x, rng)

            # Construction-level gap multiset check: exact permutation of stored gaps.
            gap_error = float(
                np.max(np.abs(np.sort(pgaps) - original_sorted_gaps))
            )
            endpoint_drift = float(abs(y[-1] - x[-1]))
            global_max_gap_multiset_error = max(global_max_gap_multiset_error, gap_error)
            global_max_endpoint_drift = max(global_max_endpoint_drift, endpoint_drift)

            if gap_error != 0.0:
                raise RuntimeError("Gap permutation failed exact multiset check.")
            if endpoint_drift > 5e-10:
                raise RuntimeError(
                    f"Unexpected surrogate endpoint drift {endpoint_drift}"
                )
            if y[0] != x[0]:
                raise RuntimeError("First point was not preserved.")

            SH[r], SV[r] = compute_profile(y, L_grid, W, n)

            if (r + 1) % 50 == 0 or (r + 1) == R:
                print(f"    completed {r+1}/{R} surrogates")

        all_surrogate_H[bi] = SH
        all_surrogate_V[bi] = SV
        all_original_H[bi] = orig_H
        all_original_V[bi] = orig_V
        roles.append(role)
        regimes.append(regime)
        block_numbers.append(block)

        # Pointwise surrogate summaries.
        Hmean = SH.mean(axis=0)
        Hsd = SH.std(axis=0, ddof=1)
        Hlo = np.quantile(SH, 0.025, axis=0)
        Hhi = np.quantile(SH, 0.975, axis=0)

        Vmean = SV.mean(axis=0)
        Vsd = SV.std(axis=0, ddof=1)
        Vlo = np.quantile(SV, 0.025, axis=0)
        Vhi = np.quantile(SV, 0.975, axis=0)

        for j, L in enumerate(L_grid):
            pointwise_rows.append({
                "role": role,
                "regime": regime,
                "block": block,
                "L": float(L),
                "H_original": float(orig_H[j]),
                "H_surrogate_mean": float(Hmean[j]),
                "H_surrogate_sd": float(Hsd[j]),
                "H_surrogate_q025": float(Hlo[j]),
                "H_surrogate_q975": float(Hhi[j]),
                "H_original_minus_surrogate_mean": float(orig_H[j] - Hmean[j]),
                "H_original_inside_95pct_band": bool(Hlo[j] <= orig_H[j] <= Hhi[j]),
                "V_original": float(orig_V[j]),
                "V_surrogate_mean": float(Vmean[j]),
                "V_surrogate_sd": float(Vsd[j]),
                "V_surrogate_q025": float(Vlo[j]),
                "V_surrogate_q975": float(Vhi[j]),
                "V_original_minus_surrogate_mean": float(orig_V[j] - Vmean[j]),
                "V_original_inside_95pct_band": bool(Vlo[j] <= orig_V[j] <= Vhi[j]),
            })

        # Original-to-surrogate-mean whole-profile distances.
        H_obs_rmse = rmse(orig_H, Hmean)
        H_obs_sup = supdist(orig_H, Hmean)
        H_jmax = int(np.argmax(np.abs(orig_H - Hmean)))

        V_obs_rmse = rmse(orig_V, Vmean)
        V_obs_sup = supdist(orig_V, Vmean)
        V_jmax = int(np.argmax(np.abs(orig_V - Vmean)))

        # Leave-one-out surrogate reference distributions.
        Htotal = SH.sum(axis=0)
        Vtotal = SV.sum(axis=0)
        Hloo = (Htotal[None, :] - SH) / float(R - 1)
        Vloo = (Vtotal[None, :] - SV) / float(R - 1)

        H_self_rmse = np.sqrt(np.mean((SH - Hloo) ** 2, axis=1))
        H_self_sup = np.max(np.abs(SH - Hloo), axis=1)
        V_self_rmse = np.sqrt(np.mean((SV - Vloo) ** 2, axis=1))
        V_self_sup = np.max(np.abs(SV - Vloo), axis=1)

        for r in range(R):
            self_rows.append({
                "role": role,
                "regime": regime,
                "block": block,
                "surrogate": r + 1,
                "H_rmse_to_LOO_surrogate_mean": float(H_self_rmse[r]),
                "H_sup_to_LOO_surrogate_mean": float(H_self_sup[r]),
                "V_rmse_to_LOO_surrogate_mean": float(V_self_rmse[r]),
                "V_sup_to_LOO_surrogate_mean": float(V_self_sup[r]),
            })

        H_inside = (orig_H >= Hlo) & (orig_H <= Hhi)
        V_inside = (orig_V >= Vlo) & (orig_V <= Vhi)

        block_rows.append({
            "role": role,
            "regime": regime,
            "block": block,
            "R_surrogate": R,

            "H_rmse_original_to_surrogate_mean": H_obs_rmse,
            "H_sup_original_to_surrogate_mean": H_obs_sup,
            "H_L_at_sup": float(L_grid[H_jmax]),
            "H_signed_difference_at_sup": float(orig_H[H_jmax] - Hmean[H_jmax]),
            "H_surrogate_LOO_rmse_percentile": empirical_percentile(H_self_rmse, H_obs_rmse),
            "H_empirical_upper_tail": empirical_upper_tail(H_self_rmse, H_obs_rmse),
            "H_fraction_L_inside_surrogate_95pct_band": float(np.mean(H_inside)),
            "H_num_L_inside_surrogate_95pct_band": int(np.sum(H_inside)),
            "H_mean_signed_difference": float(np.mean(orig_H - Hmean)),

            "V_rmse_original_to_surrogate_mean": V_obs_rmse,
            "V_sup_original_to_surrogate_mean": V_obs_sup,
            "V_L_at_sup": float(L_grid[V_jmax]),
            "V_signed_difference_at_sup": float(orig_V[V_jmax] - Vmean[V_jmax]),
            "V_surrogate_LOO_rmse_percentile": empirical_percentile(V_self_rmse, V_obs_rmse),
            "V_empirical_upper_tail": empirical_upper_tail(V_self_rmse, V_obs_rmse),
            "V_fraction_L_inside_surrogate_95pct_band": float(np.mean(V_inside)),
            "V_num_L_inside_surrogate_95pct_band": int(np.sum(V_inside)),
            "V_mean_signed_difference": float(np.mean(orig_V - Vmean)),

            "original_profile_H_recompute_max_abs_error": H_mismatch,
            "original_profile_V_recompute_max_abs_error": V_mismatch,
        })

    block_path = outdir / "surrogate_block_comparison.csv"
    pointwise_path = outdir / "surrogate_pointwise_summary.csv"
    self_path = outdir / "surrogate_self_distance_distribution.csv"

    write_csv(block_path, block_rows)
    write_csv(pointwise_path, pointwise_rows)
    write_csv(self_path, self_rows)

    # Regime-level summary over the four blocks.
    bdf = pd.DataFrame(block_rows)
    regime_rows = []
    for (role, regime), sub in bdf.groupby(["role", "regime"], sort=True):
        regime_rows.append({
            "role": role,
            "regime": regime,
            "num_blocks": int(len(sub)),
            "H_rmse_mean": float(sub["H_rmse_original_to_surrogate_mean"].mean()),
            "H_rmse_sd": float(sub["H_rmse_original_to_surrogate_mean"].std(ddof=1)),
            "H_rmse_min": float(sub["H_rmse_original_to_surrogate_mean"].min()),
            "H_rmse_max": float(sub["H_rmse_original_to_surrogate_mean"].max()),
            "H_mean_empirical_upper_tail": float(sub["H_empirical_upper_tail"].mean()),
            "H_mean_fraction_L_inside_surrogate_band": float(
                sub["H_fraction_L_inside_surrogate_95pct_band"].mean()
            ),
            "H_mean_signed_difference": float(sub["H_mean_signed_difference"].mean()),

            "V_rmse_mean": float(sub["V_rmse_original_to_surrogate_mean"].mean()),
            "V_rmse_sd": float(sub["V_rmse_original_to_surrogate_mean"].std(ddof=1)),
            "V_rmse_min": float(sub["V_rmse_original_to_surrogate_mean"].min()),
            "V_rmse_max": float(sub["V_rmse_original_to_surrogate_mean"].max()),
            "V_mean_empirical_upper_tail": float(sub["V_empirical_upper_tail"].mean()),
            "V_mean_fraction_L_inside_surrogate_band": float(
                sub["V_fraction_L_inside_surrogate_95pct_band"].mean()
            ),
            "V_mean_signed_difference": float(sub["V_mean_signed_difference"].mean()),
        })

    regime_path = outdir / "surrogate_regime_summary.csv"
    write_csv(regime_path, regime_rows)

    # Compressed full profile archive.
    npz_path = outdir / "surrogate_profiles_full.npz"
    np.savez_compressed(
        npz_path,
        L=L_grid,
        original_H=all_original_H,
        original_V=all_original_V,
        surrogate_H=all_surrogate_H,
        surrogate_V=all_surrogate_V,
        role=np.array(roles, dtype="U20"),
        regime=np.array(regimes, dtype="U30"),
        block=np.array(block_numbers, dtype=int),
    )

    manifest = {
        "purpose": "gap-shuffled surrogate experiment for count entropy and number variance",
        "surrogate_definition": "permute the empirical nearest-neighbour gaps and reconstruct from the original first point",
        "preserved_by_construction": [
            "number of points n",
            "empirical nearest-neighbour gap multiset",
            "first retained point",
            "last retained point up to floating summation roundoff",
            "observation interval [0,W]"
        ],
        "not_claimed": [
            "preservation of all two-point statistics",
            "destruction of every possible higher-order dependency",
            "formal hypothesis-test interpretation of empirical upper-tail fractions"
        ],
        "principal_block_dir": str(principal_dir.as_posix()),
        "robustness_block_dir": str(robustness_dir.as_posix()),
        "validated_original_profiles": str(original_path.as_posix()),
        "n": n,
        "W": W,
        "L_min": float(L_grid[0]),
        "L_max": float(L_grid[-1]),
        "L_step": args.L_step,
        "num_L_values": int(len(L_grid)),
        "R_surrogate_per_block": R,
        "num_blocks": len(blocks),
        "num_surrogate_profiles": int(len(blocks) * R),
        "random_seed": args.seed,
        "rng": "numpy.default_rng with SeedSequence-spawned independent block streams",
        "count_profile_algorithm": "exact breakpoint integration",
        "entropy_log_base": "natural",
        "number_variance_computed_from_same_count_pmf": True,
        "primary_reference_distance_distribution": "leave-one-out surrogate realization-to-surrogate-ensemble-mean distances within each source block",
        "empirical_upper_tail_definition": "(1 + count(surrogate_LOO_distance >= original_distance))/(R_surrogate + 1)",
        "pointwise_band": "2.5% and 97.5% surrogate quantiles at each L within each source block",
        "qc": {
            "max_original_H_recompute_abs_error": global_max_original_H_mismatch,
            "max_original_V_recompute_abs_error": global_max_original_V_mismatch,
            "max_gap_multiset_error_by_construction": global_max_gap_multiset_error,
            "max_surrogate_last_point_drift": global_max_endpoint_drift,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "outputs": {
            "block_comparison": str(block_path.as_posix()),
            "pointwise_summary": str(pointwise_path.as_posix()),
            "surrogate_self_distances": str(self_path.as_posix()),
            "regime_summary": str(regime_path.as_posix()),
            "full_profile_archive": str(npz_path.as_posix()),
        },
    }

    manifest_path = outdir / "surrogate_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("SUCCESS: gap-shuffled surrogate experiment complete.")
    print(f"Maximum original-H recomputation error: {global_max_original_H_mismatch:.3e}")
    print(f"Maximum original-V recomputation error: {global_max_original_V_mismatch:.3e}")
    print(f"Maximum gap multiset error:             {global_max_gap_multiset_error:.3e}")
    print(f"Maximum last-point drift:               {global_max_endpoint_drift:.3e}")
    print("")
    print(pd.DataFrame(regime_rows).to_string(index=False))
    print("")
    print(f"Outputs written to: {outdir.resolve()}")
    print("STOP HERE. Interpret the surrogate results before manuscript prose.")


if __name__ == "__main__":
    main()
