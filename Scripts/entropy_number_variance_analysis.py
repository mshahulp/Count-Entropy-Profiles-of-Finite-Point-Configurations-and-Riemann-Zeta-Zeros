
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TOLERANCES = (0.001, 0.0025, 0.005, 0.01, 0.02)


def massey_bound(v):
    v = np.asarray(v, dtype=float)
    return 0.5 * np.log(2.0 * np.pi * np.e * (v + 1.0 / 12.0))


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(a):
    # pandas gives average ranks for ties, sufficient for Spearman.
    return pd.Series(np.asarray(a)).rank(method="average").to_numpy(float)


def spearman(x, y):
    return pearson(rankdata_average(x), rankdata_average(y))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def observation_id(row):
    return f"{row.role}|{row.regime}|b{int(row.block):02d}|L={float(row.L):.1f}"


def load_pmf_map(path):
    df = pd.read_csv(path)
    required = {"role", "regime", "block", "L", "k", "p"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PMF file missing columns: {sorted(missing)}")

    out = {}
    for key, sub in df.groupby(["role", "regime", "block", "L"], sort=False):
        ks = sub["k"].to_numpy(int)
        ps = sub["p"].to_numpy(float)
        if np.any(ps <= 0):
            raise ValueError(f"Sparse PMF contains non-positive stored mass for {key}")
        out[(str(key[0]), str(key[1]), int(key[2]), round(float(key[3]), 10))] = (ks, ps)
    return out


def get_pmf(pmf_map, row):
    key = (str(row.role), str(row.regime), int(row.block), round(float(row.L), 10))
    if key not in pmf_map:
        raise KeyError(f"PMF not found for {key}")
    return pmf_map[key]


def pmf_metrics(ks, ps):
    s = float(ps.sum())
    mean = float(np.dot(ks, ps))
    var = float(np.dot((ks - mean) ** 2, ps))
    H = float(-np.sum(ps * np.log(ps)))
    return s, mean, var, H, int(len(ks))


def pmf_tv(ks1, p1, ks2, p2):
    lo = min(int(ks1.min()), int(ks2.min()))
    hi = max(int(ks1.max()), int(ks2.max()))
    a = np.zeros(hi - lo + 1)
    b = np.zeros(hi - lo + 1)
    a[ks1 - lo] = p1
    b[ks2 - lo] = p2
    return float(0.5 * np.sum(np.abs(a - b)))


def same_configuration(a, b):
    return (
        str(a.role) == str(b.role)
        and str(a.regime) == str(b.regime)
        and int(a.block) == int(b.block)
    )


def pair_search_same_L(df, rel_tol, abs_floor):
    """
    Search pairs at identical L from different source configurations.
    Variance closeness criterion:
      |v1-v2| <= max(abs_floor, rel_tol * max((v1+v2)/2, abs_floor))
    """
    candidates = []
    for L, sub in df.groupby("L", sort=True):
        rows = list(sub.itertuples(index=False))
        for i in range(len(rows)):
            a = rows[i]
            for j in range(i + 1, len(rows)):
                b = rows[j]
                if same_configuration(a, b):
                    continue
                v1 = float(a.number_variance)
                v2 = float(b.number_variance)
                dv = abs(v1 - v2)
                vscale = max(0.5 * (v1 + v2), abs_floor)
                threshold = max(abs_floor, rel_tol * vscale)
                if dv <= threshold:
                    dH = abs(float(a.H) - float(b.H))
                    candidates.append((dH, dv, threshold, a, b))
    candidates.sort(key=lambda z: (-z[0], z[1]))
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True,
                    help="zeta_count_entropy_profiles.csv")
    ap.add_argument("--pmf", required=True,
                    help="zeta_count_pmf_sparse.csv")
    ap.add_argument("--surrogate-npz", default=None,
                    help="optional surrogate_profiles_full.npz")
    ap.add_argument("--output-dir", default="entropy_variance_results")
    ap.add_argument("--abs-variance-floor", type=float, default=1e-4)
    ap.add_argument("--top-pairs-per-tolerance", type=int, default=20)
    ap.add_argument("--pmf-examples", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.profiles)
    required = {"role", "regime", "block", "L", "H", "number_variance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Profiles file missing columns: {sorted(missing)}")

    df = df.copy()
    df["massey_bound"] = massey_bound(df["number_variance"].to_numpy(float))
    df["massey_slack"] = df["massey_bound"] - df["H"]
    df["observation_id"] = [observation_id(r) for r in df.itertuples(index=False)]

    if np.any(~np.isfinite(df[["H", "number_variance", "massey_bound", "massey_slack"]])):
        raise ValueError("Non-finite H/variance/bound values.")
    if np.min(df["number_variance"]) < -1e-12:
        raise ValueError("Negative variance detected.")

    # Massey audit.
    min_slack = float(df["massey_slack"].min())
    violations = df[df["massey_slack"] < -1e-10].copy()
    if len(violations):
        raise RuntimeError(
            f"Massey bound violation beyond tolerance. Minimum slack={min_slack}"
        )

    # Association summaries.
    assoc_rows = []
    groups = [("all", "all", df)]
    for (role, regime), sub in df.groupby(["role", "regime"], sort=True):
        groups.append((str(role), str(regime), sub))
    for role, regime, sub in groups:
        assoc_rows.append({
            "role": role,
            "regime": regime,
            "n_observations": int(len(sub)),
            "pearson_H_vs_variance": pearson(sub["H"], sub["number_variance"]),
            "spearman_H_vs_variance": spearman(sub["H"], sub["number_variance"]),
            "H_min": float(sub["H"].min()),
            "H_max": float(sub["H"].max()),
            "variance_min": float(sub["number_variance"].min()),
            "variance_max": float(sub["number_variance"].max()),
            "massey_slack_min": float(sub["massey_slack"].min()),
            "massey_slack_median": float(sub["massey_slack"].median()),
            "massey_slack_max": float(sub["massey_slack"].max()),
        })
    assoc_path = outdir / "entropy_variance_association_summary.csv"
    write_csv(assoc_path, assoc_rows)

    # Full point table for plotting / bound inspection.
    points_cols = [
        "role", "regime", "block", "L", "H", "number_variance",
        "massey_bound", "massey_slack", "observation_id"
    ]
    points_path = outdir / "entropy_variance_points.csv"
    df[points_cols].to_csv(points_path, index=False)

    # Pair searches.
    pair_rows = []
    top_candidates_for_examples = []
    for tol in TOLERANCES:
        cands = pair_search_same_L(df, tol, args.abs_variance_floor)
        for rank, (dH, dv, threshold, a, b) in enumerate(
            cands[:args.top_pairs_per_tolerance], start=1
        ):
            meanv = 0.5 * (float(a.number_variance) + float(b.number_variance))
            rel = dv / meanv if meanv > 0 else float("nan")
            pair_rows.append({
                "relative_variance_tolerance": tol,
                "rank_within_tolerance": rank,
                "L": float(a.L),
                "obs1": observation_id(a),
                "obs2": observation_id(b),
                "role1": a.role,
                "regime1": a.regime,
                "block1": int(a.block),
                "H1": float(a.H),
                "variance1": float(a.number_variance),
                "role2": b.role,
                "regime2": b.regime,
                "block2": int(b.block),
                "H2": float(b.H),
                "variance2": float(b.number_variance),
                "abs_entropy_difference": dH,
                "abs_variance_difference": dv,
                "relative_variance_difference": rel,
                "allowed_variance_difference": threshold,
            })
        # Prefer stringent tolerances when selecting PMF examples.
        top_candidates_for_examples.extend(
            [(tol,) + c for c in cands[:max(args.pmf_examples, 10)]]
        )

    pair_path = outdir / "near_equal_variance_pairs.csv"
    write_csv(pair_path, pair_rows)

  
    pmf_map = load_pmf_map(Path(args.pmf))
    top_candidates_for_examples.sort(key=lambda z: (z[0], -z[1]))
    seen = set()
    example_rows = []
    selected = []
    for item in top_candidates_for_examples:
        tol, dH, dv, threshold, a, b = item
        ids = tuple(sorted((observation_id(a), observation_id(b))))
        if ids in seen:
            continue
        seen.add(ids)
        selected.append(item)
        if len(selected) >= args.pmf_examples:
            break

    for exrank, item in enumerate(selected, start=1):
        tol, dH, dv, threshold, a, b = item
        k1, p1 = get_pmf(pmf_map, a)
        k2, p2 = get_pmf(pmf_map, b)
        s1, m1, v1, h1, supp1 = pmf_metrics(k1, p1)
        s2, m2, v2, h2, supp2 = pmf_metrics(k2, p2)
        tv = pmf_tv(k1, p1, k2, p2)

        # Cross-check sparse PMF reconstruction against profile archive.
        if abs(v1 - float(a.number_variance)) > 5e-10 or abs(h1 - float(a.H)) > 5e-10:
            raise RuntimeError(f"PMF/profile mismatch for {observation_id(a)}")
        if abs(v2 - float(b.number_variance)) > 5e-10 or abs(h2 - float(b.H)) > 5e-10:
            raise RuntimeError(f"PMF/profile mismatch for {observation_id(b)}")

        example_rows.append({
            "example_rank": exrank,
            "selection_tolerance": tol,
            "L": float(a.L),
            "obs1": observation_id(a),
            "obs2": observation_id(b),
            "H1": h1,
            "H2": h2,
            "abs_entropy_difference": abs(h1 - h2),
            "variance1": v1,
            "variance2": v2,
            "abs_variance_difference": abs(v1 - v2),
            "mean_count1": m1,
            "mean_count2": m2,
            "support_size1": supp1,
            "support_size2": supp2,
            "pmf_total_variation_distance": tv,
            "pmf_sum1": s1,
            "pmf_sum2": s2,
        })

    examples_path = outdir / "near_equal_variance_pmf_examples.csv"
    write_csv(examples_path, example_rows)

    # Export PMF masses for selected examples for later figure/table use.
    mass_rows = []
    for exrank, item in enumerate(selected, start=1):
        tol, dH, dv, threshold, a, b = item
        for which, row in ((1, a), (2, b)):
            ks, ps = get_pmf(pmf_map, row)
            for k, p in zip(ks, ps):
                mass_rows.append({
                    "example_rank": exrank,
                    "which_observation": which,
                    "observation_id": observation_id(row),
                    "L": float(row.L),
                    "k": int(k),
                    "p": float(p),
                })
    masses_path = outdir / "selected_pmf_masses.csv"
    write_csv(masses_path, mass_rows)

    # Optional surrogate archive analysis.
    surrogate_rows = []
    if args.surrogate_npz:
        z = np.load(args.surrogate_npz)
        SH = np.asarray(z["surrogate_H"], float)
        SV = np.asarray(z["surrogate_V"], float)
        roles = z["role"].astype(str)
        regimes = z["regime"].astype(str)
        blocks = z["block"].astype(int)
        if SH.shape != SV.shape:
            raise ValueError("Surrogate H/V archive shape mismatch.")

        surrogate_rows.append({
            "scope": "all_surrogate_observations",
            "role": "all",
            "regime": "all",
            "block": "all",
            "n_observations": int(SH.size),
            "pearson_H_vs_variance": pearson(SH.ravel(), SV.ravel()),
            "spearman_H_vs_variance": spearman(SH.ravel(), SV.ravel()),
        })
        for bi in range(SH.shape[0]):
            surrogate_rows.append({
                "scope": "source_block",
                "role": roles[bi],
                "regime": regimes[bi],
                "block": int(blocks[bi]),
                "n_observations": int(SH[bi].size),
                "pearson_H_vs_variance": pearson(SH[bi].ravel(), SV[bi].ravel()),
                "spearman_H_vs_variance": spearman(SH[bi].ravel(), SV[bi].ravel()),
            })

    surrogate_path = outdir / "surrogate_entropy_variance_association.csv"
    if surrogate_rows:
        write_csv(surrogate_path, surrogate_rows)

    # Pair-search summary for quick interpretation.
    search_summary_rows = []
    for tol in TOLERANCES:
        sub = [r for r in pair_rows if r["relative_variance_tolerance"] == tol]
        search_summary_rows.append({
            "relative_variance_tolerance": tol,
            "num_top_pairs_archived": len(sub),
            "largest_entropy_difference_among_archived_pairs":
                max((r["abs_entropy_difference"] for r in sub), default=float("nan")),
            "smallest_variance_difference_among_archived_pairs":
                min((r["abs_variance_difference"] for r in sub), default=float("nan")),
        })
    search_summary_path = outdir / "near_equal_variance_search_summary.csv"
    write_csv(search_summary_path, search_summary_rows)

    manifest = {
        "purpose": "empirical entropy-number-variance relationship analysis",
        "profiles_file": str(Path(args.profiles).as_posix()),
        "sparse_pmf_file": str(Path(args.pmf).as_posix()),
        "surrogate_npz_file": str(Path(args.surrogate_npz).as_posix()) if args.surrogate_npz else None,
        "num_original_observations": int(len(df)),
        "num_original_configurations": int(df[["role", "regime", "block"]].drop_duplicates().shape[0]),
        "L_min": float(df["L"].min()),
        "L_max": float(df["L"].max()),
        "massey_bound": "0.5*log(2*pi*e*(variance+1/12))",
        "minimum_massey_slack": min_slack,
        "massey_violations_below_minus_1e-10": int(len(violations)),
        "association_metrics": ["Pearson correlation", "Spearman rank correlation"],
        "pair_search_primary_constraint": "same L and different source configurations",
        "relative_variance_tolerances": list(TOLERANCES),
        "absolute_variance_floor": args.abs_variance_floor,
        "pmf_verification": "selected candidate pairs reconstructed from sparse count-law archive and cross-checked against stored H and variance",
        "interpretation_guardrail": "correlation does not imply functional equivalence; near-equal-variance examples demonstrate only that variance alone does not determine entropy for the observed count laws",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "outputs": {
            "association_summary": str(assoc_path.as_posix()),
            "points": str(points_path.as_posix()),
            "near_equal_variance_pairs": str(pair_path.as_posix()),
            "near_equal_variance_search_summary": str(search_summary_path.as_posix()),
            "pmf_examples": str(examples_path.as_posix()),
            "selected_pmf_masses": str(masses_path.as_posix()),
            "surrogate_association": str(surrogate_path.as_posix()) if surrogate_rows else None,
        },
    }
    manifest_path = outdir / "entropy_variance_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("SUCCESS: entropy-number-variance analysis complete.")
    print(f"Original observations: {len(df)}")
    print(f"Minimum Massey slack: {min_slack:.12g}")
    print(f"Massey violations (< -1e-10): {len(violations)}")
    print("")
    print("Association summary:")
    print(pd.DataFrame(assoc_rows).to_string(index=False))
    print("")
    print("Near-equal-variance search:")
    print(pd.DataFrame(search_summary_rows).to_string(index=False))
    print("")
    if example_rows:
        print("Strongest PMF-verified examples:")
        print(pd.DataFrame(example_rows).to_string(index=False))
    else:
        print("No qualifying near-equal-variance pairs found under the specified tolerances.")
    print("")
    print(f"Outputs written to: {outdir.resolve()}")
    print("STOP HERE. Review results before manuscript prose.")


if __name__ == "__main__":
    main()
