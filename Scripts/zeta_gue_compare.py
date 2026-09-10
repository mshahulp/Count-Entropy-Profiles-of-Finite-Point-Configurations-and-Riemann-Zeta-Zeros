
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
import pandas as pd


def parse_gue_profiles(path: Path):
    df = pd.read_csv(path)
    if "realization" not in df.columns:
        raise ValueError("GUE file must contain a 'realization' column.")

    lcols = [c for c in df.columns if c.startswith("L_")]
    if not lcols:
        raise ValueError("No L_* columns found in GUE file.")

    def lval(c):
        return float(c[2:])

    lcols = sorted(lcols, key=lval)
    L = np.array([lval(c) for c in lcols], dtype=float)
    profiles = df[lcols].to_numpy(dtype=float)
    ids = df["realization"].to_numpy()
    return ids, L, profiles


def parse_zeta_profiles(path: Path):
    df = pd.read_csv(path)
    required = {"role", "regime", "block", "L", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Zeta file missing columns: {sorted(missing)}")

    keys = (
        df[["role", "regime", "block"]]
        .drop_duplicates()
        .sort_values(["role", "regime", "block"])
    )

    profiles = []
    metadata = []
    common_L = None

    for row in keys.itertuples(index=False):
        sub = df[
            (df["role"] == row.role)
            & (df["regime"] == row.regime)
            & (df["block"] == row.block)
        ].sort_values("L")

        L = sub["L"].to_numpy(dtype=float)
        H = sub["H"].to_numpy(dtype=float)

        if common_L is None:
            common_L = L
        elif not np.allclose(L, common_L, atol=1e-12, rtol=0):
            raise ValueError(f"Inconsistent zeta L grid for {row.regime} block {row.block}")

        profiles.append(H)
        metadata.append((row.role, row.regime, int(row.block)))

    return metadata, common_L, np.vstack(profiles)


def rmse_rows(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((X - mu[None, :]) ** 2, axis=1))


def sup_rows(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.max(np.abs(X - mu[None, :]), axis=1)


def empirical_upper_tail(ref: np.ndarray, d: float) -> float:
    return float((1 + np.sum(ref >= d)) / (len(ref) + 1))


def empirical_percentile(ref: np.ndarray, d: float) -> float:
    # Fraction of reference distances <= d.
    return float(100.0 * np.mean(ref <= d))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zeta-profiles",
        required=True,
        help="Path to zeta_count_entropy_profiles.csv"
    )
    parser.add_argument(
        "--gue-master-profiles",
        required=True,
        help="Path to archived gue_master_profiles.csv"
    )
    parser.add_argument("--R", type=int, default=400)
    parser.add_argument(
        "--selection",
        choices=["prefix"],
        default="prefix",
        help="Frozen rule: use first R archived realizations."
    )
    parser.add_argument("--output-dir", default="zeta_gue_comparison_results")
    args = parser.parse_args()

    zeta_path = Path(args.zeta_profiles)
    gue_path = Path(args.gue_master_profiles)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    gue_ids_all, L_gue, G_all = parse_gue_profiles(gue_path)
    zeta_meta, L_zeta, Z = parse_zeta_profiles(zeta_path)

    if G_all.shape[0] < args.R:
        raise ValueError(f"GUE archive has only {G_all.shape[0]} realizations; R={args.R} requested.")

    if args.selection == "prefix":
        gue_ids = gue_ids_all[:args.R]
        G = G_all[:args.R, :]

    if G.shape[1] != len(L_zeta):
        raise ValueError(
            f"Grid-size mismatch: GUE has {G.shape[1]} L values, zeta has {len(L_zeta)}."
        )
    if not np.allclose(L_gue, L_zeta, atol=1e-12, rtol=0):
        raise ValueError("GUE and zeta L grids do not match exactly enough for comparison.")

    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(Z)):
        raise ValueError("Non-finite entropy values detected.")

    R = G.shape[0]
    mu = G.mean(axis=0)
    sd = G.std(axis=0, ddof=1)
    q025 = np.quantile(G, 0.025, axis=0)
    q975 = np.quantile(G, 0.975, axis=0)

    # Common-mean self distances.
    gue_rmse_common = rmse_rows(G, mu)
    gue_sup_common = sup_rows(G, mu)

    # Leave-one-out means and distances.
    total = G.sum(axis=0)
    loo_means = (total[None, :] - G) / float(R - 1)
    gue_rmse_loo = np.sqrt(np.mean((G - loo_means) ** 2, axis=1))
    gue_sup_loo = np.max(np.abs(G - loo_means), axis=1)

    # Archive selected GUE pointwise summary.
    gue_summary_rows = []
    for j, L in enumerate(L_gue):
        gue_summary_rows.append({
            "L": float(L),
            "GUE_mean_R400": float(mu[j]),
            "GUE_sd_across_realizations": float(sd[j]),
            "GUE_q025": float(q025[j]),
            "GUE_q975": float(q975[j]),
        })
    gue_summary_path = outdir / "gue_selected_profile_summary.csv"
    write_csv(gue_summary_path, gue_summary_rows)

    # Archive GUE distance reference distributions.
    self_rows = []
    for i in range(R):
        self_rows.append({
            "realization": gue_ids[i],
            "rmse_to_common_GUE_mean": float(gue_rmse_common[i]),
            "sup_to_common_GUE_mean": float(gue_sup_common[i]),
            "rmse_to_leave_one_out_GUE_mean": float(gue_rmse_loo[i]),
            "sup_to_leave_one_out_GUE_mean": float(gue_sup_loo[i]),
        })
    self_path = outdir / "gue_self_distance_distribution.csv"
    write_csv(self_path, self_rows)

    # Zeta block distances and pointwise coverage.
    block_rows = []
    coverage_rows = []

    for i, (role, regime, block) in enumerate(zeta_meta):
        z = Z[i]
        diff = z - mu
        d_rmse = float(np.sqrt(np.mean(diff ** 2)))
        d_sup = float(np.max(np.abs(diff)))
        absdiff = np.abs(diff)
        jmax = int(np.argmax(absdiff))

        within = (z >= q025) & (z <= q975)
        frac_within = float(np.mean(within))
        longest_run = 0
        current = 0
        for flag in within:
            if flag:
                current += 1
                longest_run = max(longest_run, current)
            else:
                current = 0

        block_rows.append({
            "role": role,
            "regime": regime,
            "block": block,
            "rmse_to_GUE_mean": d_rmse,
            "sup_to_GUE_mean": d_sup,
            "L_at_sup_difference": float(L_gue[jmax]),
            "signed_difference_at_sup": float(diff[jmax]),
            "rmse_GUE_LOO_percentile": empirical_percentile(gue_rmse_loo, d_rmse),
            "rmse_empirical_upper_tail": empirical_upper_tail(gue_rmse_loo, d_rmse),
            "sup_GUE_LOO_percentile": empirical_percentile(gue_sup_loo, d_sup),
            "sup_empirical_upper_tail": empirical_upper_tail(gue_sup_loo, d_sup),
            "fraction_L_inside_GUE_95pct_pointwise_band": frac_within,
            "num_L_inside_GUE_95pct_pointwise_band": int(np.sum(within)),
            "longest_consecutive_L_run_inside_band": int(longest_run),
        })

        coverage_rows.append({
            "role": role,
            "regime": regime,
            "block": block,
            "fraction_L_inside_GUE_95pct_pointwise_band": frac_within,
            "num_L_inside_GUE_95pct_pointwise_band": int(np.sum(within)),
            "num_L_below_GUE_q025": int(np.sum(z < q025)),
            "num_L_above_GUE_q975": int(np.sum(z > q975)),
        })

    block_path = outdir / "zeta_gue_block_distances.csv"
    write_csv(block_path, block_rows)

    coverage_path = outdir / "zeta_gue_pointwise_band_coverage.csv"
    write_csv(coverage_path, coverage_rows)

    # Regime-level descriptive summaries from the four zeta blocks.
    block_df = pd.DataFrame(block_rows)
    regime_rows = []
    for (role, regime), sub in block_df.groupby(["role", "regime"], sort=True):
        regime_rows.append({
            "role": role,
            "regime": regime,
            "num_zeta_blocks": int(len(sub)),
            "rmse_mean": float(sub["rmse_to_GUE_mean"].mean()),
            "rmse_sd": float(sub["rmse_to_GUE_mean"].std(ddof=1)),
            "rmse_min": float(sub["rmse_to_GUE_mean"].min()),
            "rmse_max": float(sub["rmse_to_GUE_mean"].max()),
            "sup_mean": float(sub["sup_to_GUE_mean"].mean()),
            "sup_sd": float(sub["sup_to_GUE_mean"].std(ddof=1)),
            "mean_fraction_L_inside_GUE_95pct_pointwise_band": float(
                sub["fraction_L_inside_GUE_95pct_pointwise_band"].mean()
            ),
            "min_fraction_L_inside_GUE_95pct_pointwise_band": float(
                sub["fraction_L_inside_GUE_95pct_pointwise_band"].min()
            ),
            "max_fraction_L_inside_GUE_95pct_pointwise_band": float(
                sub["fraction_L_inside_GUE_95pct_pointwise_band"].max()
            ),
        })
    regime_path = outdir / "zeta_gue_regime_distance_summary.csv"
    write_csv(regime_path, regime_rows)

    manifest = {
        "purpose": "quantitative zeta-GUE count-entropy profile comparison",
        "zeta_profiles_file": str(zeta_path.as_posix()),
        "gue_master_profiles_file": str(gue_path.as_posix()),
        "gue_archive_realizations_available": int(G_all.shape[0]),
        "frozen_principal_GUE_R": int(R),
        "gue_selection_rule": "first R archived realizations",
        "L_min": float(L_gue[0]),
        "L_max": float(L_gue[-1]),
        "num_L_values": int(len(L_gue)),
        "num_zeta_profiles": int(Z.shape[0]),
        "rmse_definition": "sqrt(mean_L((H_profile-H_GUE_mean)^2))",
        "sup_definition": "max_L(abs(H_profile-H_GUE_mean))",
        "primary_GUE_reference_distance_distribution": "leave-one-out realization-to-ensemble-mean distances",
        "empirical_upper_tail_definition": "(1 + count(GUE_LOO_distance >= zeta_distance))/(R + 1)",
        "empirical_upper_tail_interpretation": "descriptive reference-tail measure, not labeled as a formal hypothesis-test p-value",
        "pointwise_band": "2.5% and 97.5% quantiles across the selected R GUE realizations at each L",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "outputs": {
            "gue_summary": str(gue_summary_path.as_posix()),
            "gue_self_distances": str(self_path.as_posix()),
            "zeta_block_distances": str(block_path.as_posix()),
            "zeta_regime_summary": str(regime_path.as_posix()),
            "zeta_band_coverage": str(coverage_path.as_posix()),
        },
    }

    manifest_path = outdir / "zeta_gue_comparison_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("SUCCESS: zeta-GUE quantitative comparison complete.")
    print(f"Selected GUE realizations: first {R} of {G_all.shape[0]} archived master realizations.")
    print(f"GUE L grid: {L_gue[0]} ... {L_gue[-1]} ({len(L_gue)} values)")
    print("")
    print("GUE leave-one-out RMSE reference:")
    for q in (0.50, 0.90, 0.95, 0.975, 0.99):
        print(f"  q{100*q:5.1f}: {np.quantile(gue_rmse_loo, q):.6f}")
    print("")
    print("Regime summary:")
    print(pd.DataFrame(regime_rows).to_string(index=False))
    print("")
    print(f"Outputs written to: {outdir.resolve()}")
    print("STOP HERE. Review the numerical comparison before manuscript prose.")


if __name__ == "__main__":
    main()
