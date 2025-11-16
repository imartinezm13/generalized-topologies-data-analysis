# Generalized Topologies for Semantic Similarity Detection

This repository contains code and resources for analyzing and constructing generalized topologies to study semantic similarity across entities, using multidimensional data such as economic tier, government regime, education scores, and classic machine-learning benchmarks.

## Repository Structure

- `data/`  
  - `PISA_test_database.xlsm` – main PISA country dataset.
- `results/`  
  - `PISA/` – PISA outputs (economy, government, density, continent) and sensitivity runs.  
  - `Gapminder/` – Gapminder outputs (continent, income, development) and sensitivity runs.  
  - `Synthetic/` – synthetic experiments (blobs/moons/circles) and sensitivity runs.  
  - `UCI-*/` – UCI datasets (e.g. `UCI-iris`, `UCI-wine`) and their sensitivity runs.
- `utils/` – core utilities and modules  
  - `analysis.py` – closure/interior computation and batch Excel export.  
  - `data_utils.py` – preprocessing utilities (scaling, averaging).  
  - `regression.py`, `regression_robust.py` – thresholding logic and robust knee detection.  
  - `topo.py` – generalized topology operators (closure/interior).  
  - `bases.py` – construction of `base`, `gen_base_1`, `gen_base_2`, k-means bases.  
  - `datasets.py` – loaders for PISA, Gapminder (via plotly), UCI, and synthetic datasets.  
  - `groups.py` – helpers to build categorical and quantile-based seed groups.
- `runners/` – dataset-specific and analysis entry points  
  - `gapminder_runner.py` – Gapminder pipeline.  
  - `synthetic_runner.py` – synthetic blobs/moons/circles pipeline.  
  - `uci_runner.py` – UCI (iris, wine, breast_cancer) pipeline.  
  - `sensitivity_runner.py` – sensitivity/stability sweeps (linkage, metric, noise, seeds).  
  - `sensitivity_summary.py` – aggregates sensitivity results, writes tables/plots, prints highlights.  
  - `make_tables.py` – generates LaTeX tables from sensitivity summaries.
- `main.py` – PISA pipeline and CLI for the paper experiments.  
- `requirements.txt` – Python dependencies.  
- `LICENSE`

## Description

We study the behavior of different topological bases over similarity spaces derived from multidimensional indicators. We define:

- **Standard topologies**:  
  - $B_n$: distance-based fixed-radius base.  
  - $B_{\text{k-means}}$: clusters using k-means in embedding space.
- **Generalized topologies**:  
  - $B_{A, v_{\kappa_1}}$, $B_{A, v_{\kappa_2}}$: data-aware constructions that adjust the neighborhood based on reconstruction accuracy and a regression-based threshold.

Key operations:
- **Closure**: how far the base expands from a seed set $A$.  
- **Interior**: which elements of $A$ are fully contained under a base.

## How to Run (PISA – Paper Baseline)

1. Install dependencies (preferably in a clean virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the PISA analysis (paper baseline):

   ```bash
   python main.py
   ```

   Optional CLI arguments (defaults reproduce the paper configuration):

   ```bash
   python main.py \
     --linkage single \
     --metric euclidean \
     --target-r2 0.99 \
     --n-clusters 4 \
     --random-state 42 \
     --out results/PISA
   ```

This script processes each defined seed set (e.g., South America, High Economy, Full Democracy) and computes closure/interior across all bases, exporting Excel files under `results/PISA/...`.

## Data

- **PISA** – `data/PISA_test_database.xlsm`, containing quantitative and categorical attributes for a set of countries (e.g. education scores, economy type, population density, government type, continent).  
- **Gapminder** – loaded from `plotly.express.data.gapminder()`, providing life expectancy, GDP per capita, population, continent, etc.  
- **Synthetic** – generated via scikit-learn utilities (`blobs`, `moons`, `circles`) with additional synthetic labels.  
- **UCI** – classic benchmarks (Iris, Wine, Breast Cancer) loaded via `sklearn.datasets`.

## Output

The `results/` directory contains exported closure/interior summaries by dimension (e.g. economy, government) and by base. For example:

- `results/PISA/economy/*.xlsx`  
- `results/PISA/government/*.xlsx`  
- `results/PISA/density/*.xlsx`  
- `results/PISA/continent/*.xlsx`

Additional datasets write under their own subfolders:

- `results/Gapminder/<family>/*.xlsx`  
- `results/Synthetic/<Kind>/<family>/*.xlsx`  
- `results/UCI/<name>/<family>/*.xlsx`

This mirrors the manual analysis workflow: build bases, pick a seed group, compute closure/interior, and inspect Excel outputs.

## Additional Datasets and Runners

All dataset runners follow the same pattern: build bases, define seed families, compute closure/interior, and export Excel for manual inspection.

### Gapminder — `runners/gapminder_runner.py`

- Loads the latest-year snapshot from `plotly.express.data.gapminder()`.  
- Builds `base`, `gen_base_1`, `gen_base_2`, and a k-means clustering on numeric features.  
- Seed families:
  - `continent` (categorical),
  - `income` (gdpPercap quartiles),
  - `development` (composite of lifeExp + gdpPercap).
- Outputs: `results/Gapminder/<family>/*.xlsx`.

Run:

```bash
python runners/gapminder_runner.py
```

### Synthetic — `runners/synthetic_runner.py`

- Generates small synthetic datasets (default ~80 points) of type:
  - `blobs`, `moons`, or `circles`.  
- Adds labels:
  - `cluster_true`: generator cluster labels,  
  - `regime`: orthogonal label defined by a linear rule over features,  
  - `Country`: synthetic IDs for readable exports.  
- Seed families:
  - `cluster_true` (ground truth clusters),
  - `regime` (cross-cluster label).
- Outputs: `results/Synthetic/<Kind>/<family>/*.xlsx`.

Run (default blobs):

```bash
python runners/synthetic_runner.py
```

### UCI — `runners/uci_runner.py`

- Supports `iris`, `wine`, and `breast_cancer` via `sklearn.datasets`.  
- Subsamples to at most 100 points, roughly balanced across labels (per run).  
- Seed families:
  - `label` (class family),
  - `<feature>_quantiles` (quantile-binned first numeric feature).  
- Outputs: `results/UCI/<name>/<family>/*.xlsx`.

Run:

```bash
python runners/uci_runner.py --name iris
```

## Sensitivity and Stability Analysis

To study sensitivity to linkage, distance metric, noise, and k-means seeds—and stability of the bases—we provide a unified sensitivity runner and summarizer.

### Sensitivity sweeps — `runners/sensitivity_runner.py`

Runs multi-configuration experiments and writes:

- `summary.csv` – closure/interior sizes, expansion/retention ratios, and (for Synthetic/UCI) semantic purity.  
- `stability_ari.csv` – median/mean Adjusted Rand Index across k-means seeds.  
- `stability_jaccard.csv` – median Jaccard overlap of closures across seeds.

Concretely, it varies:

- **Linkage method** – e.g. `single`, `complete`, `average`, `ward`.  
- **Distance metric** – e.g. `euclidean`, `cityblock`, `cosine`, `correlation` (for non-ward).  
- **Noise level** – small Gaussian perturbations to numeric features (testing robustness to measurement error).  
- **Random seed** – different k-means initializations (and, for Synthetic, different generations).

The analysis tests:

- How closure and interior sizes change under these perturbations (expansion vs. retention).  
- How stable the induced clusters are (ARI across seeds).  
- How stable closures are for each seed group (Jaccard overlap across runs).  
- For Synthetic and UCI label families, how well bases preserve semantic labels (closure/interior purity), and whether generalized bases remain both robust and semantically aligned compared to standard bases and k-means.

Examples:

```bash
# PISA
python runners/sensitivity_runner.py --dataset pisa

# Gapminder
python runners/sensitivity_runner.py --dataset gapminder

# Synthetic (blobs by default, 80 points)
python runners/sensitivity_runner.py --dataset synthetic

# UCI Iris (subsampled to ≤100 points)
python runners/sensitivity_runner.py --dataset uci --uci-name iris
```

### Summarizing and plotting — `runners/sensitivity_summary.py`

Aggregates sensitivity results and produces:

- `medians_per_seed.csv` – median closure/interior/ratios (and purity when available) per dataset/family/seed/base.  
- `medians_by_base.csv` – medians per dataset/family/base.  
- Plots under `summary/plots/`:
  - median closure/interior/expansion/retention (all datasets),  
  - median purity (Synthetic + UCI label families),  
  - ARI and Jaccard stability curves.
- Console highlights:
  - **Synthetic** – which bases achieve highest closure/interior purity for `cluster_true` and `regime`.  
  - **UCI** – which bases best preserve class labels in closure/interior.  
  - **PISA/Gapminder** – which bases maximize retention vs. expansion per semantic family.

Example (UCI Iris):

```bash
python runners/sensitivity_summary.py --summary results/UCI-iris/sensitivity/summary.csv
```

Replace the summary path with:

- `results/PISA/sensitivity/summary.csv` for PISA,  
- `results/Gapminder/sensitivity/summary.csv` for Gapminder,  
- `results/Synthetic/sensitivity/summary.csv` for Synthetic.

### LaTeX tables — `runners/make_tables.py`

To include compact quantitative summaries in the paper, you can generate LaTeX tables directly from a `summary.csv`:

```bash
# Example: PISA
python runners/make_tables.py --summary results/PISA/sensitivity/summary.csv

# Example: UCI Iris
python runners/make_tables.py --summary results/UCI-iris/sensitivity/summary.csv
```

This creates a `tables.tex` file next to the summary, containing one table per `(dataset, family)` pair. Each table reports, for every base (standard and generalized):

- median closure and interior sizes,  
- expansion and retention ratios,  
- and, where available (Synthetic + UCI label families), closure and interior purity with respect to labels/regimes.

These tables can be `\input{...}` into your LaTeX manuscript to support the qualitative discussion of expansion/retention, semantic coherence, and stability across datasets.

## Paper

This repository supports the article:

> *“Generalized Topologies in Data Analysis: A Flexible Approach for Classification and Similarity Detection”*  
> Julián Castañeda, Carlos Giraldo, Isabella Martínez, Margot Salas-Brown

If you use this code or data, please cite accordingly.

### Global sensitivity tables — `runners/global_sensitivity_tables.py`

To obtain a single, concise view of quality and stability across all datasets, you can run:

```bash
python runners/global_sensitivity_tables.py > results/global_sensitivity_tables.tex
```

This prints LaTeX tables that aggregate, for PISA, Gapminder, Synthetic, and UCI-Iris:

- median expansion and retention ratios per base and dataset,  
- median closure purity (where labels/regimes are available),  
- and median Jaccard and ARI stability measures across seeds and perturbations.
