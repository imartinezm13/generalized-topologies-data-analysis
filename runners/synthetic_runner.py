import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path when running directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.datasets import make_synthetic
from utils.data_utils import stand
from utils.bases import build_base, build_gen_base_1, build_gen_base_2, kmeans_clusters_from_columns
from utils.analysis import process_batch


def prepare_synthetic(
    kind: str = "blobs",
    n_samples: int = 80,
    noise: float = 0.1,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset with cluster and regime labels.

    - kind: 'blobs', 'moons', or 'circles'.
    - Adds:
      - 'cluster_true': ground-truth cluster labels from the generator.
      - 'regime': an orthogonal label induced by a linear rule on features.
      - 'Country': synthetic ID used for compatibility with process_batch.
    """

    X, y_cluster = make_synthetic(kind=kind, n_samples=n_samples, noise=noise, random_state=random_state)

    df = X.copy()
    df["cluster_true"] = y_cluster.astype(int)

    # Define an orthogonal "regime" label via a simple linear rule
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    w = rng.normal(size=X.shape[1])
    scores = X.values @ w
    threshold = np.median(scores)
    regime = (scores >= threshold).astype(int)
    df["regime"] = regime

    # Synthetic identifier for readability in Excel outputs
    df["Country"] = [f"P{i}" for i in range(len(df))]

    return df


def define_groups_synthetic(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """Build seed group families for synthetic data.

    Families:
    - 'cluster_true': groups by ground-truth cluster.
    - 'regime': groups by regime label.
    """

    families: Dict[str, Dict[str, List[str]]] = {}

    if "cluster_true" in df.columns:
        cluster_groups: Dict[str, List[str]] = {}
        for label in sorted(df["cluster_true"].unique()):
            idxs = df.index[df["cluster_true"] == label].tolist()
            cluster_groups[f"cluster_{int(label)}"] = [str(i) for i in idxs]
        families["cluster_true"] = cluster_groups

    if "regime" in df.columns:
        regime_groups: Dict[str, List[str]] = {}
        for label in sorted(df["regime"].unique()):
            idxs = df.index[df["regime"] == label].tolist()
            regime_groups[f"regime_{int(label)}"] = [str(i) for i in idxs]
        families["regime"] = regime_groups

    return families


def compute_bases_synthetic(
    df_all: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    target_r2: float = 0.99,
    n_clusters: int = 4,
    random_state: int | None = 42,
    show_progress: bool = False,
):
    """Build Base, generalized bases, and k-means clusters for synthetic data."""

    # Use only feature columns (exclude labels and synthetic ID)
    drop_cols = [c for c in ["cluster_true", "regime", "Country"] if c in df_all.columns]
    df_num = df_all.drop(columns=drop_cols)

    df_scaled = stand(df_num)

    Base = build_base(df_scaled, linkage_method=linkage_method, distance_metric=distance_metric)
    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=distance_metric, target_r2=target_r2)
    gen_base_2 = build_gen_base_2(Base, df_scaled, df_num, distance_metric=distance_metric, target_r2=target_r2, show_progress=show_progress)

    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=random_state)

    return Base, gen_base_1, gen_base_2, clusters


def run_synthetic(
    kind: str = "blobs",
    n_samples: int = 80,
    noise: float = 0.1,
    random_state: int | None = 42,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    target_r2: float = 0.99,
    n_clusters: int = 4,
    kmeans_random_state: int | None = 42,
    results_root: str = "results/Synthetic",
    show_progress: bool = True,
) -> None:
    """Run the synthetic generalized topology pipeline for a chosen shape.

    Outputs Excel files for each family under results/Synthetic/<kind>/<family>/.
    """

    df_all = prepare_synthetic(kind=kind, n_samples=n_samples, noise=noise, random_state=random_state)

    Base, gen_base_1, gen_base_2, clusters = compute_bases_synthetic(
        df_all,
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        target_r2=target_r2,
        n_clusters=n_clusters,
        random_state=kmeans_random_state,
        show_progress=show_progress,
    )

    target_groups = {
        "gen_base_1": gen_base_1,
        "gen_base_2": gen_base_2,
        "base": Base,
        "clusters": clusters,
    }

    A = [str(i) for i in range(len(df_all))]
    families = define_groups_synthetic(df_all)

    base_out_root = os.path.join(results_root, kind.capitalize())

    for fam_name, fam_groups in families.items():
        out_dir = os.path.join(base_out_root, fam_name)
        os.makedirs(out_dir, exist_ok=True)
        process_batch(fam_groups, df_all, target_groups, A, output_folder=out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic generalized topology analysis.")
    parser.add_argument("--kind", default="blobs", choices=["blobs", "moons", "circles"], help="Synthetic dataset kind")
    parser.add_argument("--n-samples", type=int, default=80, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level for synthetic generator")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--linkage", default="single", choices=["single", "complete", "average", "ward"], help="Hierarchical linkage method")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cityblock", "cosine", "correlation"], help="Distance metric (ignored for ward)")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R^2 for regression thresholding")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of clusters for k-means base")
    parser.add_argument("--kmeans-random-state", type=int, default=42, help="Random seed for k-means")
    parser.add_argument("--out", default="results/Synthetic", help="Output root directory")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar for gen_base_2")

    args = parser.parse_args()

    run_synthetic(
        kind=args.kind,
        n_samples=args.n_samples,
        noise=args.noise,
        random_state=args.random_state,
        linkage_method=args.linkage,
        distance_metric=args.metric,
        target_r2=args.target_r2,
        n_clusters=args.n_clusters,
        kmeans_random_state=args.kmeans_random_state,
        results_root=args.out,
        show_progress=not args.no_progress,
    )
