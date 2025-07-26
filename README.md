# Generalized Topologies for Semantic Similarity Detection

This repository contains code and resources for analyzing and constructing generalized topologies to study semantic similarity across countries, using multidimensional data such as economic tier, government regime, and education scores.

## Repository Structure

```

generalized-topologies-data-analysis/
├── data/                    # Raw input files
│   └── PISA\_test\_database.xlsm
├── results/                 # Output of experiments, by attribute
│   ├── continent/
│   ├── density/
│   ├── economy/
│   └── government/
├── utils/                   # Core utilities and modules
│   ├── analysis.py          # Closure/interior computation
│   ├── data\_utils.py        # Preprocessing and loading functions
│   ├── regression.py        # Thresholding logic for base construction
│   └── topo.py              # Generalized topology operators
├── main.py                  # Entry point to run batch experiments
├── LICENSE
└── requirements.txt         # Python dependencies

````

## Description

We study the behavior of different topological bases over a country similarity space derived from multidimensional indicators. We define:

- **Standard topologies**: 
  - $B_n$: distance-based fixed radius base.
  - $B_{k-means}$: clusters using k-means in embedding space.
- **Generalized topologies**: 
  - $B_{A, v_{\kappa_1}}$, $B_{A, v_{\kappa_2}}$: data-aware constructions that adjust the neighborhood based on reconstruction accuracy and a regression-based threshold.

Key operations:
- **Closure**: how far the base expands from a seed set `A`.
- **Interior**: which elements of `A` are “fully contained” under a base.

## How to Run

1. Install dependencies (preferably in a clean virtual environment):
   ```bash
   pip install -r requirements.txt
    ```

2. Run the main analysis:

   ```bash
   python main.py
   ```

This script processes each defined seed set (e.g., South America, High Economy, Full Democracy) and computes closure/interior across all bases.

## Data

We rely on the `PISA_test_database.xlsm` file (under `data/`) as the main input source, containing quantitative and categorical attributes for a set of countries.

## Output

The `results/` directory contains the exported closure/interior summaries by dimension (e.g., economy, government) and by base. These are used to support figures and tables in the accompanying paper.

## Paper

This repository supports the article:

> *“Generalized Topologies in Data Analysis: A Flexible Approach for Classification and Similarity Detection”*
> Julian Castañeda, Carlos Giraldo, Isabella Martínez, Margot Salas-Brown

If you use this code or data, please cite accordingly.