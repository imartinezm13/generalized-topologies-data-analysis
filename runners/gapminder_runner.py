import os
import sys
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path when running this file directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.datasets import load_gapminder_pkg
from utils.data_utils import stand
from utils.analysis import process_batch
from utils.groups import build_groups_from_categorical, build_groups_from_quantiles, composite_index
from utils.bases import build_base, build_gen_base_1, build_gen_base_2, kmeans_clusters_from_columns


def prepare_gapminder_latest() -> pd.DataFrame:
    """Load gapminder and return the latest-year snapshot with integer index."""
    gm = load_gapminder_pkg()
    latest_year = int(gm['year'].max())
    gm_latest = gm[gm['year'] == latest_year].reset_index(drop=True)
    return gm_latest


def compute_bases(df_all: pd.DataFrame,
                  linkage_method: str = 'single',
                  distance_metric: str = 'euclidean',
                  target_r2: float = 0.99,
                  n_clusters: int = 4,
                  random_state: int | None = 42,
                  show_progress: bool = False):
    """Build Base, gen_base_1, gen_base_2, and clusters from the provided DataFrame."""
    # Select numeric feature columns only, excluding identifiers and constant year snapshot
    numeric_df = df_all.select_dtypes(include=[np.number]).copy()
    drop_ids = [c for c in ['year', 'iso_num'] if c in numeric_df.columns]
    num_df = numeric_df.drop(columns=drop_ids)
    df_scaled = stand(num_df)

    Base = build_base(df_scaled, linkage_method=linkage_method, distance_metric=distance_metric)
    gen_base_1 = build_gen_base_1(Base, num_df, distance_metric=distance_metric, target_r2=target_r2)
    gen_base_2 = build_gen_base_2(Base, df_scaled, num_df, distance_metric=distance_metric, target_r2=target_r2, show_progress=show_progress)

    # Clusters: use all numeric columns
    feature_cols = list(num_df.columns)
    clusters = kmeans_clusters_from_columns(num_df, feature_cols, n_clusters=n_clusters, random_state=random_state)
    return Base, gen_base_1, gen_base_2, clusters


def define_groups_gapminder(df_all: pd.DataFrame):
    """Build seed groups for Gapminder: continent (categorical), income (gdpPercap quantiles), development (composite)."""
    groups = {}
    if 'continent' in df_all.columns:
        groups['continent'] = build_groups_from_categorical(df_all, 'continent')
    if 'gdpPercap' in df_all.columns:
        groups['income'] = build_groups_from_quantiles(df_all, 'gdpPercap', q=4, labels=['Low','MedLow','MedHigh','High'])
    # Composite development index from lifeExp and gdpPercap if available
    comp_cols = [c for c in ['lifeExp', 'gdpPercap'] if c in df_all.columns]
    if comp_cols:
        dev = composite_index(df_all, comp_cols, name='development')
        tmp = df_all.copy()
        tmp['development'] = dev
        groups['development'] = build_groups_from_quantiles(tmp, 'development', q=4, labels=['Low','MedLow','MedHigh','High'])
    return groups


def run_gapminder(linkage_method: str = 'single', distance_metric: str = 'euclidean', target_r2: float = 0.99, results_root: str = 'results/Gapminder', n_clusters: int = 4, random_state: int | None = 42, show_progress: bool = True):
    os.makedirs(results_root, exist_ok=True)
    df_all = prepare_gapminder_latest()
    print(df_all.head())
    Base, gen_base_1, gen_base_2, clusters = compute_bases(
        df_all,
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        target_r2=target_r2,
        n_clusters=n_clusters,
        random_state=random_state,
        show_progress=show_progress,
    )

    # Target groups for closure/interior
    target_groups = {
        'gen_base_1': gen_base_1,
        'gen_base_2': gen_base_2,
        'base': Base,
        'clusters': clusters,
    }

    # Universe A
    A = [str(i) for i in range(len(df_all))]

    # Build seed families
    families = define_groups_gapminder(df_all)

    # Process each family separately, mirroring PISA exports
    for fam_name, fam_groups in families.items():
        out_dir = os.path.join(results_root, fam_name)
        os.makedirs(out_dir, exist_ok=True)
        process_batch(fam_groups, df_all.rename(columns={'country':'Country'}), target_groups, A, output_folder=out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Gapminder generalized topology analysis.')
    parser.add_argument('--linkage', default='single', choices=['single', 'complete', 'average', 'ward'], help='Hierarchical linkage method')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cityblock', 'cosine', 'correlation'], help='Distance metric (ignored for ward)')
    parser.add_argument('--target-r2', type=float, default=0.99, help='Target R^2 for regression thresholding')
    parser.add_argument('--n-clusters', type=int, default=4, help='Number of clusters for k-means base')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for k-means')
    parser.add_argument('--out', default='results/Gapminder', help='Output root directory')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar for gen_base_2')

    args = parser.parse_args()

    run_gapminder(
        linkage_method=args.linkage,
        distance_metric=args.metric,
        target_r2=args.target_r2,
        results_root=args.out,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        show_progress=not args.no_progress,
    )
