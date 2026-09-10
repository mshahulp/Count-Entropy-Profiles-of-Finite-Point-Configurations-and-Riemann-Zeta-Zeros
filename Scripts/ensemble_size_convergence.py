

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import eigh_tridiagonal


@dataclass
class Config:
    n: int = 2000
    parent_factor: float = 2.0
    l_min: float = 0.1
    l_max: float = 20.0
    delta_l: float = 0.1
    r_master: int = 500
    candidate_r: tuple = (25, 50, 100, 200, 300, 400)
    resamples: int = 200
    seed: int = 20260829
    output_dir: str = "ensemble_R_results"


def binary_entropy_from_probs(p: np.ndarray) -> float:
    """Shannon entropy in nats for a probability vector."""
    q = p[p > 0.0]
    return float(-np.sum(q * np.log(q)))


def exact_count_entropy(points: np.ndarray, a: float, b: float, L: float) -> float:

    W = b - a
    if not (0.0 < L < W):
        raise ValueError(f"L must satisfy 0 < L < W; got L={L}, W={W}")

    right_origin = b - L

    lefts = np.maximum(a, points - L)
    rights = np.minimum(right_origin, points)

    valid = lefts < rights
    lefts = lefts[valid]
    rights = rights[valid]

    # entering a contribution interval increases count; leaving decreases.
    positions = np.concatenate((lefts, rights))
    deltas = np.concatenate(
        (np.ones(lefts.size, dtype=np.int64),
         -np.ones(rights.size, dtype=np.int64))
    )

    if positions.size == 0:
        return 0.0

    order = np.argsort(positions, kind="mergesort")
    positions = positions[order]
    deltas = deltas[order]

    # Group coincident event locations.
    unique_pos, first_idx = np.unique(positions, return_index=True)
    grouped_delta = np.add.reduceat(deltas, first_idx)

    lengths_by_count = np.zeros(points.size + 1, dtype=float)
    count = 0
    prev = a

    for pos, delta in zip(unique_pos, grouped_delta):
        if pos > prev:
            lengths_by_count[count] += pos - prev
        count += int(delta)
        prev = pos

    if right_origin > prev:
        lengths_by_count[count] += right_origin - prev

    probs = lengths_by_count / (W - L)

    # Floating-point guard.
    probs[np.abs(probs) < 1e-15] = 0.0
    probs = probs / probs.sum()

    return binary_entropy_from_probs(probs)


def entropy_profile(points: np.ndarray, a: float, b: float, L_grid: np.ndarray) -> np.ndarray:
    return np.array([exact_count_entropy(points, a, b, float(L)) for L in L_grid])


def poisson_configuration(n: int, rng: np.random.Generator):
   
    points = np.sort(rng.uniform(0.0, float(n), size=n))
    return points, 0.0, float(n)


def semicircle_cdf(x: np.ndarray) -> np.ndarray:
    """CDF of rho_sc(x)=sqrt(4-x^2)/(2*pi) on [-2,2]."""
    x = np.asarray(x, dtype=float)
    y = np.clip(x, -2.0, 2.0)
    root = np.sqrt(np.maximum(0.0, 4.0 - y * y))
    cdf = 0.5 + (y * root) / (4.0 * np.pi) + np.arcsin(y / 2.0) / np.pi
    cdf = np.where(x <= -2.0, 0.0, cdf)
    cdf = np.where(x >= 2.0, 1.0, cdf)
    return cdf


def gue_configuration(n: int, parent_factor: float, rng: np.random.Generator):
   
    N = int(round(parent_factor * n))
    if N < n + 2:
        raise ValueError("Parent GUE dimension must exceed n by at least 2.")

    diag = rng.normal(loc=0.0, scale=1.0, size=N)
    # offdiag[j] corresponds to chi with df = 2*(N-j-1), j=0,...,N-2
    dfs = 2.0 * np.arange(N - 1, 0, -1)
    offdiag = np.sqrt(rng.chisquare(df=dfs)) / np.sqrt(2.0)

    start = (N - n) // 2
    stop = start + n - 1

    # Need one discarded neighbour on each side: indices start-1 ... stop+1.
    lo = start - 1
    hi = stop + 1
    evals = eigh_tridiagonal(
        diag,
        offdiag,
        select="i",
        select_range=(lo, hi),
        check_finite=False,
        lapack_driver="auto",
    )[0]

    evals = evals / np.sqrt(float(N))
    unfolded = float(N) * semicircle_cdf(evals)

    left_neighbour = unfolded[0]
    retained = unfolded[1:-1]
    right_neighbour = unfolded[-1]

    a = 0.5 * (left_neighbour + retained[0])
    b = 0.5 * (retained[-1] + right_neighbour)

    # One affine map for points and interval; this preserves all relative geometry.
    scale = float(n) / (b - a)
    points = (retained - a) * scale
    a_new = 0.0
    b_new = float(n)

    return points, a_new, b_new


def generate_master_profiles(
    model: str,
    cfg: Config,
    L_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    profiles = np.empty((cfg.r_master, L_grid.size), dtype=float)

    for r in range(cfg.r_master):
        if model == "poisson":
            points, a, b = poisson_configuration(cfg.n, rng)
        elif model == "gue":
            points, a, b = gue_configuration(cfg.n, cfg.parent_factor, rng)
        else:
            raise ValueError(model)

        profiles[r] = entropy_profile(points, a, b, L_grid)

        if (r + 1) % max(1, cfg.r_master // 10) == 0:
            print(f"{model}: completed {r+1}/{cfg.r_master} realizations")

    return profiles


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def sup_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))


def convergence_from_master(
    profiles: np.ndarray,
    candidate_r,
    resamples: int,
    rng: np.random.Generator,
    model: str,
) -> pd.DataFrame:
    
    Rmax = profiles.shape[0]
    ref_mean = profiles.mean(axis=0)
    ref_lo = np.quantile(profiles, 0.025, axis=0)
    ref_hi = np.quantile(profiles, 0.975, axis=0)

    rows = []

    for R in candidate_r:
        if R >= Rmax:
            continue

        vals = {
            "mean_sup": [],
            "mean_rmse": [],
            "lo_sup": [],
            "lo_rmse": [],
            "hi_sup": [],
            "hi_rmse": [],
            "max_mcse": [],
            "rms_mcse": [],
        }

        for _ in range(resamples):
            idx = rng.choice(Rmax, size=R, replace=False)
            sub = profiles[idx]

            m = sub.mean(axis=0)
            lo = np.quantile(sub, 0.025, axis=0)
            hi = np.quantile(sub, 0.975, axis=0)
            mcse = sub.std(axis=0, ddof=1) / np.sqrt(float(R))

            vals["mean_sup"].append(sup_error(m, ref_mean))
            vals["mean_rmse"].append(rmse(m, ref_mean))
            vals["lo_sup"].append(sup_error(lo, ref_lo))
            vals["lo_rmse"].append(rmse(lo, ref_lo))
            vals["hi_sup"].append(sup_error(hi, ref_hi))
            vals["hi_rmse"].append(rmse(hi, ref_hi))
            vals["max_mcse"].append(float(np.max(mcse)))
            vals["rms_mcse"].append(float(np.sqrt(np.mean(mcse ** 2))))

        row = {"model": model, "R": R}
        for key, arr in vals.items():
            arr = np.asarray(arr)
            row[f"{key}_median"] = float(np.median(arr))
            row[f"{key}_p95"] = float(np.quantile(arr, 0.95))
        rows.append(row)

    return pd.DataFrame(rows)


def prefix_doubling_diagnostic(profiles: np.ndarray, candidate_r, model: str) -> pd.DataFrame:
   
    rows = []
    Rmax = profiles.shape[0]
    for R in candidate_r:
        if 2 * R > Rmax:
            continue
        m1 = profiles[:R].mean(axis=0)
        m2 = profiles[: 2 * R].mean(axis=0)
        rows.append(
            {
                "model": model,
                "R": R,
                "2R": 2 * R,
                "mean_sup_R_vs_2R": sup_error(m1, m2),
                "mean_rmse_R_vs_2R": rmse(m1, m2),
            }
        )
    return pd.DataFrame(rows)


def save_profile_summary(profiles: np.ndarray, L_grid: np.ndarray, model: str, outdir: Path):
    df = pd.DataFrame(
        {
            "L": L_grid,
            "mean": profiles.mean(axis=0),
            "sd": profiles.std(axis=0, ddof=1),
            "q025": np.quantile(profiles, 0.025, axis=0),
            "q975": np.quantile(profiles, 0.975, axis=0),
            "mcse_mean": profiles.std(axis=0, ddof=1) / np.sqrt(profiles.shape[0]),
        }
    )
    df.to_csv(outdir / f"{model}_master_profile_summary.csv", index=False)

    # Raw profiles are important for audit/re-analysis.
    raw = pd.DataFrame(profiles, columns=[f"L_{x:.6g}" for x in L_grid])
    raw.insert(0, "realization", np.arange(1, profiles.shape[0] + 1))
    raw.to_csv(outdir / f"{model}_master_profiles.csv", index=False)


def make_convergence_plot(df: pd.DataFrame, model: str, outdir: Path):
    sub = df[df["model"] == model].sort_values("R")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sub["R"], sub["mean_rmse_p95"], marker="o", label="Mean profile: RMS error (95th pct.)")
    ax.plot(sub["R"], sub["lo_rmse_p95"], marker="o", label="Lower band: RMS error (95th pct.)")
    ax.plot(sub["R"], sub["hi_rmse_p95"], marker="o", label="Upper band: RMS error (95th pct.)")
    ax.set_xlabel("Candidate ensemble size R")
    ax.set_ylabel("RMS discrepancy from master ensemble (nats)")
    ax.set_title(f"{model.upper()} ensemble-size convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{model}_R_convergence.png", dpi=180)
    plt.close(fig)


def environment_metadata(cfg: Config):
    return {
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "timestamp_unix": time.time(),
        "notes": {
            "entropy_units": "nats (natural logarithm)",
            "window_origin_sampling": "none; exact breakpoint/event integration",
            "poisson_control": "n iid Uniform(0,n) points; fixed cardinality",
            "gue_model": "beta-Hermite tridiagonal, beta=2",
            "gue_unfolding": "semicircle CDF followed by affine W=n normalization",
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Run a small pipeline sanity check.")
    p.add_argument("--r-master", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.quick:
        cfg.n = 120
        cfg.parent_factor = 2.0
        cfg.l_min = 0.5
        cfg.l_max = 4.0
        cfg.delta_l = 0.5
        cfg.r_master = 20
        cfg.candidate_r = (5, 10)
        cfg.resamples = 30
        cfg.output_dir = "ensemble_R_quick_test"

    if args.r_master is not None:
        cfg.r_master = args.r_master
    if args.n is not None:
        cfg.n = args.n
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    candidate_r = tuple(r for r in cfg.candidate_r if r < cfg.r_master)
    if not candidate_r:
        raise ValueError("At least one candidate R must be smaller than r_master.")

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Avoid cumulative floating-point endpoint drift from arange.
    count = int(round((cfg.l_max - cfg.l_min) / cfg.delta_l)) + 1
    L_grid = np.linspace(cfg.l_min, cfg.l_max, count)

    # Independent reproducible RNG streams for the two models and for resampling.
    seed_seq = np.random.SeedSequence(cfg.seed)
    ss_poisson, ss_gue, ss_resample_p, ss_resample_g = seed_seq.spawn(4)

    rng_poisson = np.random.default_rng(ss_poisson)
    rng_gue = np.random.default_rng(ss_gue)
    rng_resample_p = np.random.default_rng(ss_resample_p)
    rng_resample_g = np.random.default_rng(ss_resample_g)

    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(environment_metadata(cfg), f, indent=2)

    print("Generating Poisson master ensemble...")
    poisson_profiles = generate_master_profiles("poisson", cfg, L_grid, rng_poisson)
    save_profile_summary(poisson_profiles, L_grid, "poisson", outdir)

    print("Generating GUE master ensemble...")
    gue_profiles = generate_master_profiles("gue", cfg, L_grid, rng_gue)
    save_profile_summary(gue_profiles, L_grid, "gue", outdir)

    print("Assessing candidate R values...")
    conv_p = convergence_from_master(
        poisson_profiles, candidate_r, cfg.resamples, rng_resample_p, "poisson"
    )
    conv_g = convergence_from_master(
        gue_profiles, candidate_r, cfg.resamples, rng_resample_g, "gue"
    )
    conv = pd.concat([conv_p, conv_g], ignore_index=True)
    conv.to_csv(outdir / "R_convergence_metrics.csv", index=False)

    prefix = pd.concat(
        [
            prefix_doubling_diagnostic(poisson_profiles, candidate_r, "poisson"),
            prefix_doubling_diagnostic(gue_profiles, candidate_r, "gue"),
        ],
        ignore_index=True,
    )
    prefix.to_csv(outdir / "R_prefix_doubling_check.csv", index=False)

    make_convergence_plot(conv, "poisson", outdir)
    make_convergence_plot(conv, "gue", outdir)

    print("\nCandidate-R convergence summary:")
    cols = [
        "model",
        "R",
        "mean_sup_p95",
        "mean_rmse_p95",
        "lo_rmse_p95",
        "hi_rmse_p95",
        "max_mcse_p95",
    ]
    print(conv[cols].to_string(index=False))
    print(f"\nSaved reproducibility outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
