

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import mpmath as mp
import numpy as np
import requests


ODLYZKO_PAGE = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    url: str
    global_first_zero_number: int
    parent_count: int = 10_000
    stated_accuracy: str = ""


DATASETS = (
    DatasetSpec(
        key="low",
        label="first_10000",
        url=ODLYZKO_PAGE + "zeros1",
        global_first_zero_number=1,
        parent_count=10_000,
        stated_accuracy="within 3e-9 (source statement for first 100,000 zeros)",
    ),
    DatasetSpec(
        key="1e12",
        label="near_1e12",
        url=ODLYZKO_PAGE + "zeros3",
        global_first_zero_number=10**12 + 1,
        parent_count=10_000,
        stated_accuracy="within 1e-8 (source guarantee)",
    ),
    DatasetSpec(
        key="1e21",
        label="near_1e21",
        url=ODLYZKO_PAGE + "zeros4",
        global_first_zero_number=10**21 + 1,
        parent_count=10_000,
        stated_accuracy="probably within 1e-6; source says not guaranteed",
    ),
    DatasetSpec(
        key="1e22",
        label="near_1e22",
        url=ODLYZKO_PAGE + "zeros5",
        global_first_zero_number=10**22 + 1,
        parent_count=10_000,
        stated_accuracy="probably within 1e-6; source says not guaranteed",
    ),
)


@dataclass
class Config:
    n: int = 2000
    block_starts: tuple[int, ...] = (1, 2001, 4001, 6001)
    mp_dps: int = 80
    output_dir: str = "zeta_prepared"
    timeout_seconds: int = 60


NUMERIC_LINE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$")
OFFSET_HEADER = re.compile(r"Values\s+of\s+gamma\s*-\s*([0-9]+)", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_text(url: str, timeout: int) -> tuple[str, bytes]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    raw = response.content
    # Odlyzko tables are plain ASCII.
    return raw.decode("ascii"), raw


def parse_odlyzko_table(text: str) -> tuple[list[mp.mpf], mp.mpf, dict]:
 
    lines = text.splitlines()

    base = mp.mpf("0")
    offset_mode = False

    for line in lines[:12]:
        m = OFFSET_HEADER.search(line)
        if m:
            base = mp.mpf(m.group(1))
            offset_mode = True
            break

    numeric_tokens: list[str] = []
    for line in lines:
        m = NUMERIC_LINE.match(line)
        if m:
            numeric_tokens.append(m.group(1))

    # Header lines in offset files do not match NUMERIC_LINE, so these are data.
    vals = [mp.mpf(tok) for tok in numeric_tokens]
    if offset_mode:
        vals = [base + v for v in vals]

    return vals, base, {
        "offset_mode": offset_mode,
        "parsed_base": mp.nstr(base, 50),
        "numeric_lines": len(numeric_tokens),
    }


def nbar(T: mp.mpf) -> mp.mpf:
    two_pi = 2 * mp.pi
    return (T / two_pi) * mp.log(T / two_pi) - (T / two_pi) + mp.mpf(7) / 8


def unfold_locally(gammas: list[mp.mpf]) -> list[mp.mpf]:
  
    origin = nbar(gammas[0])
    return [nbar(g) - origin for g in gammas]


def make_block(
    parent_gammas: list[mp.mpf],
    start: int,
    n: int,
) -> dict:
    
    if start < 1:
        raise ValueError("start must be >= 1 so a left guard exists.")
    if start + n >= len(parent_gammas):
        raise ValueError("Not enough points for retained block plus right guard.")

    local_gamma = parent_gammas[start - 1 : start + n + 1]  # n + 2 points
    if len(local_gamma) != n + 2:
        raise RuntimeError("Unexpected block length.")

    # Strict monotonicity at source precision.
    gamma_gaps = [local_gamma[i + 1] - local_gamma[i] for i in range(len(local_gamma) - 1)]
    if not all(g > 0 for g in gamma_gaps):
        raise ValueError("Source ordinates are not strictly increasing.")

    unfolded = unfold_locally(local_gamma)

    left_guard = unfolded[0]
    retained = unfolded[1:-1]
    right_guard = unfolded[-1]

    a = (left_guard + retained[0]) / 2
    b = (retained[-1] + right_guard) / 2
    W_pre = b - a

    scale = mp.mpf(n) / W_pre
    normalized_mp = [(x - a) * scale for x in retained]

    # After the high-precision local normalization the values are O(n), so float64

    normalized64 = np.array([float(x) for x in normalized_mp], dtype=np.float64)

    if not np.all(np.diff(normalized64) > 0):
        raise ValueError("Monotonicity lost after float64 conversion.")

    final_gaps = np.diff(normalized64)

    # Quantify round-trip error caused by the final float64 conversion only.
    max_float64_error = max(
        abs(mp.mpf(float(y)) - y) for y in normalized_mp
    )

    return {
        "normalized_mp": normalized_mp,
        "normalized64": normalized64,
        "raw_gamma_gaps": gamma_gaps,
        "unfolded": unfolded,
        "a": a,
        "b": b,
        "W_pre": W_pre,
        "scale": scale,
        "final_gaps": final_gaps,
        "max_float64_error": max_float64_error,
        "first_gamma": local_gamma[1],
        "last_gamma": local_gamma[-2],
        "left_guard_gamma": local_gamma[0],
        "right_guard_gamma": local_gamma[-1],
    }


def naive_float64_unique_fraction(full_gammas: list[mp.mpf]) -> float:
    """
    Diagnostic only: fraction of distinct values after naively converting the
    full high-height ordinates to float64. This shows why high precision is needed.
    """
    arr = np.array([float(x) for x in full_gammas], dtype=np.float64)
    return float(np.unique(arr).size / arr.size)


def mp_to_str(x: mp.mpf, digits: int = 30) -> str:
    return mp.nstr(x, digits)


def save_block_csv(path: Path, normalized64: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "normalized_unfolded_coordinate"])
        for i, x in enumerate(normalized64, start=1):
            writer.writerow([i, format(float(x), ".17g")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="zeta_prepared")
    parser.add_argument("--mp-dps", type=int, default=80)
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use already-downloaded raw files in output_dir/raw instead of downloading.",
    )
    args = parser.parse_args()

    cfg = Config(mp_dps=args.mp_dps, output_dir=args.output_dir)
    mp.mp.dps = cfg.mp_dps

    outdir = Path(cfg.output_dir)
    rawdir = outdir / "raw"
    blockdir = outdir / "blocks"
    rawdir.mkdir(parents=True, exist_ok=True)
    blockdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_page": ODLYZKO_PAGE,
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "requests": requests.__version__,
        "datasets": [],
        "unfolding": "Nbar(T)=T/(2*pi)*log(T/(2*pi))-T/(2*pi)+7/8",
        "oscillatory_S_T_included": False,
        "final_interval": "[0,n]",
        "entropy_computed_here": False,
    }

    validation_rows = []

    for spec in DATASETS:
        raw_path = rawdir / f"{spec.key}.txt"

        if args.use_existing:
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"{raw_path} does not exist. Run without --use-existing first."
                )
            raw = raw_path.read_bytes()
            text = raw.decode("ascii")
        else:
            print(f"Downloading {spec.key}: {spec.url}")
            text, raw = download_text(spec.url, cfg.timeout_seconds)
            raw_path.write_bytes(raw)

        digest = sha256_bytes(raw)

        gammas, parsed_base, parse_info = parse_odlyzko_table(text)

        if len(gammas) < spec.parent_count:
            raise ValueError(
                f"{spec.key}: parsed {len(gammas)} zeros; expected at least {spec.parent_count}."
            )

        parent = gammas[: spec.parent_count]

        if not all(parent[i + 1] > parent[i] for i in range(len(parent) - 1)):
            raise ValueError(f"{spec.key}: parent sequence is not strictly increasing.")

        naive_unique = naive_float64_unique_fraction(parent)

        manifest["datasets"].append(
            {
                "key": spec.key,
                "label": spec.label,
                "url": spec.url,
                "raw_file": str(raw_path.as_posix()),
                "sha256": digest,
                "global_first_zero_number": str(spec.global_first_zero_number),
                "parent_count": spec.parent_count,
                "stated_accuracy": spec.stated_accuracy,
                **parse_info,
            }
        )

        for block_no, start in enumerate(cfg.block_starts, start=1):
            result = make_block(parent, start, cfg.n)

            global_start = spec.global_first_zero_number + start
            global_end = global_start + cfg.n - 1

            block_name = f"{spec.key}_block{block_no:02d}.csv"
            save_block_csv(blockdir / block_name, result["normalized64"])

            gaps = result["final_gaps"]
            raw_gaps = result["raw_gamma_gaps"][1:-1]  # retained-to-retained gaps
            raw_gap_float = np.array([float(g) for g in raw_gaps], dtype=float)

            validation_rows.append(
                {
                    "dataset": spec.key,
                    "block": block_no,
                    "global_zero_start": str(global_start),
                    "global_zero_end": str(global_end),
                    "retained_n": cfg.n,
                    "source_monotone": True,
                    "naive_full_gamma_float64_unique_fraction": naive_unique,
                    "first_gamma": mp_to_str(result["first_gamma"], 40),
                    "last_gamma": mp_to_str(result["last_gamma"], 40),
                    "raw_mean_gap_gamma": float(np.mean(raw_gap_float)),
                    "raw_min_gap_gamma": float(np.min(raw_gap_float)),
                    "raw_max_gap_gamma": float(np.max(raw_gap_float)),
                    "pre_affine_W": float(result["W_pre"]),
                    "affine_scale": float(result["scale"]),
                    "final_W": float(cfg.n),
                    "final_intensity_n_over_W": 1.0,
                    "final_mean_nn_gap": float(np.mean(gaps)),
                    "final_min_nn_gap": float(np.min(gaps)),
                    "final_max_nn_gap": float(np.max(gaps)),
                    "final_coordinate_min": float(result["normalized64"][0]),
                    "final_coordinate_max": float(result["normalized64"][-1]),
                    "float64_monotone_after_normalization": bool(np.all(gaps > 0)),
                    "max_abs_error_from_final_float64_conversion": float(result["max_float64_error"]),
                    "mp_dps": cfg.mp_dps,
                    "block_file": str((blockdir / block_name).as_posix()),
                }
            )

    validation_path = outdir / "zeta_validation.csv"
    fieldnames = list(validation_rows[0].keys())
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_rows)

    with (outdir / "zeta_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nValidation complete.")
    print(f"16 block files: {blockdir.resolve()}")
    print(f"Validation table: {validation_path.resolve()}")
    print(f"Manifest: {(outdir / 'zeta_manifest.json').resolve()}")
    print("\nSTOP HERE before entropy computation.")
    print("Inspect zeta_validation.csv and zeta_manifest.json first.")


if __name__ == "__main__":
    main()
