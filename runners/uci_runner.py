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

from utils.datasets import load_uci
from utils.data_utils import stand
from utils.bases import build_base, build_gen_base_1, build_gen_base_2, kmeans_clusters_from_columns
from utils.groups import build_groups_from_quantiles
from utils.analysis import process_batch


def _balanced_subsample_indices(labels: pd.Series, max_n: int, random_state: int | None = None) -> np.ndarray:
    """Return indices for a roughly balanced subsample across classes, up to max_n total."""
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    y = labels.to_numpy()
    unique = np.unique(y)
    n_classes = len(unique)
    if n_classes == 0 or max_n <= 0:
        return np.arange(len(labels))
    per_class = max_n // n_classes
    if per_class == 0:
        per_class = 1
    chosen: list[np.ndarray] = []
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) <= per_class:
            sel = cls_idx
        else:
            sel = rng.choice(cls_idx, size=per_class, replace=False)
        chosen.append(sel)
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx


def prepare_uci(name: str, max_samples: int | None = 100, random_state: int | None = 42) -> pd.DataFrame:
    """Load a UCI dataset and return a DataFrame with features, label, and synthetic IDs.

    If max_samples is set and the dataset is larger, subsample up to max_samples
    points in a roughly class-balanced way.
    """
    X, y = load_uci(name)
    if max_samples is not None and len(X) > max_samples:
        idx = _balanced_subsample_indices(y, max_samples, random_state=random_state)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
    df = X.copy()
    df["label"] = y.values
    df["Country"] = [f"P{i}" for i in range(len(df))]
    return df


def define_groups_uci(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """Build seed families for UCI data: class labels and a quantile-binned feature."""
    families: Dict[str, Dict[str, List[str]]] = {}

    # Label family
    if "label" in df.columns:
        label_groups: Dict[str, List[str]] = {}
        for val in sorted(df["label"].unique()):
            idxs = df.index[df["label"] == val].tolist()
            label_groups[f"class_{int(val)}"] = [str(i) for i in idxs]
        families["label"] = label_groups

    # Quantile-binned first numeric feature (excluding label)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != "label"]
    if feature_cols:
        col = feature_cols[0]
        fam_name = f"{col}_quantiles"
        families[fam_name] = build_groups_from_quantiles(
            df, col, q=4, labels=["Low", "MedLow", "MedHigh", "High"]
        )

    return families


def compute_bases_uci(
    df_all: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    target_r2: float = 0.99,
    n_clusters: int = 4,
    random_state: int | None = 42,
    show_progress: bool = False,
):
    """Build Base, generalized bases, and k-means clusters for UCI data."""
    df_num = df_all.drop(columns=[c for c in ["label", "Country"] if c in df_all.columns])
    df_scaled = stand(df_num)

    Base = build_base(df_scaled, linkage_method=linkage_method, distance_metric=distance_metric)
    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=distance_metric, target_r2=target_r2)
    gen_base_2 = build_gen_base_2(
        Base,
        df_scaled,
        df_num,
        distance_metric=distance_metric,
        target_r2=target_r2,
        show_progress=show_progress,
    )
    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=random_state)
    return Base, gen_base_1, gen_base_2, clusters


def run_uci(
    name: str = "iris",
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    target_r2: float = 0.99,
    n_clusters: int = 4,
    random_state: int | None = 42,
    results_root: str = "results/UCI",
    show_progress: bool = True,
    max_samples: int | None = 100,
) -> None:
    """Run the UCI generalized topology pipeline for a given dataset."""
    df_all = prepare_uci(name, max_samples=max_samples, random_state=random_state)

    Base, gen_base_1, gen_base_2, clusters = compute_bases_uci(
        df_all,
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        target_r2=target_r2,
        n_clusters=n_clusters,
        random_state=random_state,
        show_progress=show_progress,
    )

    target_groups = {
        "gen_base_1": gen_base_1,
        "gen_base_2": gen_base_2,
        "base": Base,
        "clusters": clusters,
    }
    A = [str(i) for i in range(len(df_all))]
    families = define_groups_uci(df_all)

    base_out_root = os.path.join(results_root, name)

    for fam_name, fam_groups in families.items():
        out_dir = os.path.join(base_out_root, fam_name)
        os.makedirs(out_dir, exist_ok=True)
        process_batch(fam_groups, df_all, target_groups, A, output_folder=out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UCI generalized topology analysis.")
    parser.add_argument("--name", default="iris", choices=["iris", "wine", "breast_cancer"], help="UCI dataset name")
    parser.add_argument("--linkage", default="single", choices=["single", "complete", "average", "ward"], help="Hierarchical linkage method")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cityblock", "cosine", "correlation"], help="Distance metric (ignored for ward)")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R^2 for regression thresholding")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of clusters for k-means base")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for k-means")
    parser.add_argument("--out", default="results/UCI", help="Output root directory")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar for gen_base_2")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of samples to use (balanced across classes)")

    args = parser.parse_args()

    run_uci(
        name=args.name,
        linkage_method=args.linkage,
        distance_metric=args.metric,
        target_r2=args.target_r2,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        results_root=args.out,
        show_progress=not args.no_progress,
        max_samples=args.max_samples,
    )
