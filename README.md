# Count-Entropy Analysis of Riemann-Zeta Zero Configurations

## Companion Code Repository

This repository contains the complete computational pipeline and figure-generation scripts for the manuscript:

**"Sliding-Window Count Entropy of Riemann-Zeta Zero Configurations"**

The code is provided to accompany the manuscript submission and enables full reproduction of all numerical results and publication figures.

---

## Overview

This study investigates the Shannon count entropy of Riemann-zeta zero configurations under a sliding-window observation model, and compares the resulting empirical profiles against predictions from random matrix theory (the Gaussian Unitary Ensemble, GUE) and a Poisson baseline. The analysis demonstrates that count-entropy profiles of zeta zeros are statistically indistinguishable from the GUE reference across a wide range of spectral heights, from the first 10^4 zeros up to heights near 10^22.

---

## Repository Structure

```

├── README.md                               # This file
├── Final_Figures.py                        # Master figure-generation script (Figures 1–5)
│
└── Run code/                               # Analysis and computation scripts
    ├── zeta_prepare_validate.py            # Stage 1: Download, unfold, and normalize zeta zero blocks
    ├── zeta_lowheight_robustness.py        # Stage 1b: Intermediate-low robustness dataset
    ├── zeta_compute_entropy.py             # Stage 2: Exact count-entropy computation for all blocks
    ├── ensemble_size_convergence.py        # Stage 3: GUE/Poisson ensemble-size convergence study
    ├── zeta_gue_compare.py                 # Stage 4: Quantitative zeta-to-GUE comparison
    ├── gap_shuffled_surrogates.py          # Stage 5: Gap-shuffled surrogate experiment
    └── entropy_number_variance_analysis.py # Stage 6: Entropy–number-variance relationship analysis
```

---

## Computational Pipeline

The analysis proceeds in six sequential stages. Each stage reads validated outputs from the previous stage, enabling independent auditing and verification.

### Stage 1: Zeta Zero Preparation and Validation (`zeta_prepare_validate.py`)

Downloads Riemann-zeta zero ordinate tables from the Odlyzko archive and constructs normalized zero blocks:

- **Low height**: First 10^4 zeros from `zeros1`
- **Near 10^12**: Zeros 10^12+1 to 10^12+10^4 from `zeros3`
- **Near 10^21**: Zeros 10^21+1 to 10^21+10^4 from `zeros4`
- **Near 10^22**: Zeros 10^22+1 to 10^22+10^4 from `zeros5`

Each 10,000-zero parent yields four non-overlapping blocks of n=2,000 retained zeros with guard-point boundary normalization. High-precision mpmath arithmetic (80 decimal digits) is used for the high-height datasets.

**Outputs**: 16 principal block CSV files, validation table, manifest.

### Stage 1b: Intermediate-Low Robustness Blocks (`zeta_lowheight_robustness.py`)

Constructs an additional four blocks from zeros 45,001–55,000 (centred near zero 50,000) to test robustness of results at intermediate low height. Uses identical unfolding and normalization rules.

**Outputs**: 4 robustness block CSV files, validation table, manifest.

### Stage 2: Exact Count-Entropy Computation (`zeta_compute_entropy.py`)

Computes the exact sliding-window count-entropy profile H(L) for L in {0.1, 0.2, …, 20.0} for all 20 zeta blocks (16 principal + 4 robustness).

- **Algorithm**: Exact breakpoint partition with midpoint counts via `numpy.searchsorted`
- **Cross-check**: Independent event-sweep integration at selected L values
- **QC conditions**: PMF normalization, non-negativity, entropy bounds, variance sign

**Outputs**: Entropy profiles, QC report, sparse PMF archive, regime-level summaries, manifest.

### Stage 3: Ensemble-Size Convergence Study (`ensemble_size_convergence.py`)

Determines a defensible ensemble size R for the GUE and Poisson reference ensembles by:

1. Generating a master ensemble of R_master=500 count-entropy profiles
2. Comparing candidate sizes R = {25, 50, 100, 200, 300, 400} against the master
3. Quantifying convergence via RMSE and supremum error of the mean profile, quantile bands, and Monte Carlo standard error

**Key result**: R=400 selected as the frozen principal GUE ensemble size.

**Outputs**: Master profile archives, convergence metrics, diagnostic plots, manifest.

### Stage 4: Zeta-to-GUE Comparison (`zeta_gue_compare.py`)

Quantitative comparison of validated zeta entropy profiles against the archived GUE master ensemble (R=400):

- **Metrics**: Profile-wise RMSE and supremum discrepancy to the GUE ensemble mean
- **Reference distribution**: GUE leave-one-out self-distance distribution
- **Empirical upper-tail fractions**: Descriptive reference-tail measure at each zeta block
- **Pointwise coverage**: Fraction of L values within the GUE 95% pointwise band

**Outputs**: Block-distance table, regime summary, band-coverage report, GUE profile summary, manifest.

### Stage 5: Gap-Shuffled Surrogate Experiment (`gap_shuffled_surrogates.py`)

Tests whether count entropy can distinguish a zeta configuration from surrogates that preserve the exact nearest-neighbour gap multiset but randomise gap ordering. For each block, 200 independent gap permutations are generated and evaluated.

**Key properties**: Preserves n, gap multiset, first point, observation interval. Randomises sequential gap order.

**Outputs**: Block comparison table, pointwise summary, self-distance distribution, full profile archive (compressed), manifest.

### Stage 6: Entropy–Number-Variance Analysis (`entropy_number_variance_analysis.py`)

Empirically examines the relationship between Shannon count entropy H(L) and number variance Σ²(L):

- Global and within-regime association (Pearson and Spearman correlations)
- Massey integer-valued entropy bound audit
- Near-equal-variance / different-entropy pair search across configurations
- PMF verification of strongest candidate examples

**Outputs**: Association summary, points table, pair-search results, PMF examples, manifest.

---

## Figure Generation

### Master Script

All five publication figures can be generated with:

```bash
python Final_Figures.py
```

Individual figures can be selected:

```bash
python Final_Figures.py --figures 3
python Final_Figures.py --figures 1 4 5
```

The script reads the pre-computed CSV outputs produced by the pipeline stages
(see the reproduction commands below) and writes SVG, PDF, and PNG (600 dpi)
outputs into per-figure output directories.

### Figure Descriptions

| Figure | Description |
|--------|-------------|
| 1 | Reference count-entropy profiles (GUE, Poisson, lattice) with ensemble-mean count laws |
| 2 | Zeta count-entropy profiles across spectral height with consecutive-regime displacement and pairwise RMS distances |
| 3 | GUE leave-one-out calibration ECDF and joint profile discrepancy for zeta configurations |
| 4 | Gap-shuffled surrogate analysis: scale-resolved displacement, configuration–scale heatmap, within-surrogate calibration |
| 5 | Empirical entropy–variance relation, maximum entropy separation under variance matching, and PMF comparison |

All figures are defined as independent functions within `Final_Figures.py`
(`generate_figure1` through `generate_figure5`, dispatched by the master
`main()` entry point) and are exported in SVG, PDF, and PNG (600 dpi) formats.

---

## Requirements

### Python Version

Python >= 3.10

### Dependencies

```
numpy
scipy
pandas
matplotlib
mpmath
requests
```

Install all dependencies:

```bash
pip install numpy scipy pandas matplotlib mpmath requests
```

---

## Reproducing the Full Analysis

```bash
# Stage 1: Download and prepare zeta zero blocks
python Run\ code/zeta_prepare_validate.py

# Stage 1b: Prepare intermediate-low robustness blocks
python Run\ code/zeta_lowheight_robustness.py

# Stage 2: Compute exact count-entropy profiles
python Run\ code/zeta_compute_entropy.py

# Stage 3: Ensemble-size convergence study
python Run\ code/ensemble_size_convergence.py

# Stage 4: Zeta-to-GUE quantitative comparison
python Run\ code/zeta_gue_compare.py \
    --zeta-profiles zeta_entropy_results/zeta_count_entropy_profiles.csv \
    --gue-master-profiles ensemble_R_results/gue_master_profiles.csv

# Stage 5: Gap-shuffled surrogate experiment
python Run\ code/gap_shuffled_surrogates.py \
    --principal-block-dir zeta_prepared/blocks \
    --robustness-block-dir zeta_lowheight_robustness/blocks \
    --original-profiles zeta_entropy_results/zeta_count_entropy_profiles.csv

# Stage 6: Entropy-number-variance analysis
python Run\ code/entropy_number_variance_analysis.py \
    --profiles zeta_entropy_results/zeta_count_entropy_profiles.csv \
    --pmf zeta_entropy_results/zeta_count_pmf_sparse.csv

# Generate all publication figures
python Final_Figures.py
```

---

## Key Methodological Notes

1. **Exact computation**: All count-entropy and count-PMF values are computed exactly (up to floating-point arithmetic) via breakpoint integration; no Monte Carlo sampling of window origins is used.

2. **High-precision arithmetic**: Zeta zero ordinate parsing at heights 10^21–10^22 uses mpmath with 80 decimal digits to avoid catastrophic cancellation.

3. **Ensemble selection**: The GUE ensemble size R=400 was frozen after a convergence study against a master ensemble of R=500.

4. **Descriptive statistics**: Empirical upper-tail fractions are reported as reference-tail diagnostics, not as formal hypothesis-test p-values.

5. **Entropy units**: All entropies are in nats (natural logarithm base).

6. **Random seeds**: All random number generation uses `numpy.random.Generator` with explicit `SeedSequence` seeds for full reproducibility.

---

## Data Sources

Riemann-zeta zero ordinate tables are from:

> Andrew M. Odlyzko, *Tables of zeros of the Riemann zeta function*
> https://www-users.cse.umn.edu/~odlyzko/zeta_tables/

---

## License

This code is provided as supplementary material for the manuscript. Please cite appropriately if using any part of this code or its outputs.

---

## Contact

For questions regarding the code or data, please contact the corresponding author listed in the manuscript.
