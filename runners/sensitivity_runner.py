import os
import sys
import json
from itertools import product
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Path shim for repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Force non-interactive matplotlib backend for any downstream imports
os.environ.setdefault('MPLBACKEND', 'Agg')

from utils.analysis import refine_base
from utils.topo import closure, interior
from utils.data_utils import stand
from utils.bases import build_base, build_gen_base_1, build_gen_base_2, kmeans_clusters_from_columns
from utils.datasets import make_synthetic, load_uci

# PISA imports
from main import load_data as load_pisa
from main import define_groups_and_targets as pisa_define_groups

# Gapminder imports
from runners.gapminder_runner import prepare_gapminder_latest, define_groups_gapminder

# Synthetic imports
from runners.synthetic_runner import define_groups_synthetic

# UCI imports
from runners.uci_runner import define_groups_uci


def add_noise(df_num: pd.DataFrame, sigma: float, rng: np.random.RandomState) -> pd.DataFrame:
    if sigma <= 0:
        return df_num.copy()
    noisy = df_num.copy()
    for col in noisy.columns:
        std = float(noisy[col].std(ddof=0))
        if std == 0:
            continue
        noise = rng.normal(loc=0.0, scale=sigma * std, size=len(noisy))
        noisy[col] = noisy[col].astype(float) + noise
    return noisy


def summarize_family(df_display: pd.DataFrame,
                     A: List[str],
                     families: Dict[str, Dict[str, List[str]]],
                     target_groups: Dict[str, List[List[str]]]) -> List[Dict]:
    """Return summary rows for closure/interior sizes across families and bases."""
    rows = []
    for fam_name, fam_groups in families.items():
        for seed_name, base_group in fam_groups.items():
            # refine gen bases with respect to this base group
            gen_base_1_re = refine_base(base_group, target_groups["gen_base_1"], target_groups["base"])
            gen_base_2_re = refine_base(base_group, target_groups["gen_base_2"], target_groups["base"])
            tg_ref = {
                "gen_base_1": gen_base_1_re,
                "gen_base_2": gen_base_2_re,
                "base": target_groups["base"],
                "clusters": target_groups["clusters"],
            }
            for base_name, Base in tg_ref.items():
                clos = [int(i) for i in closure(Base, base_group, A)]
                inte = [int(i) for i in interior(Base, base_group)]
                rows.append({
                    "family": fam_name,
                    "seed": seed_name,
                    "base": base_name,
                    "closure_size": len(clos),
                    "interior_size": len(inte),
                    "seed_size": len(base_group),
                    "n_total": len(A),
                    # Store closures/interiors as compact strings for stability/quality metrics
                    "closure_indices": ",".join(map(str, sorted(clos))),
                    "interior_indices": ",".join(map(str, sorted(inte))),
                })
    return rows


def _labels_from_groups(groups: list[list[str]], n: int) -> np.ndarray:
    labels = np.full(n, fill_value=-1, dtype=int)
    for cid, idxs in enumerate(groups):
        for s in idxs:
            labels[int(s)] = cid
    return labels


def _balanced_subsample_indices_array(labels: np.ndarray, max_n: int, random_state: int | None = None) -> np.ndarray:
    """Balanced subsample helper for sensitivity runs."""
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    y = np.asarray(labels)
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


def _pairwise_jaccard(sets: list[set[int]]) -> float:
    if len(sets) < 2:
        return float('nan')
    vals = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            inter = len(sets[i] & sets[j])
            uni = len(sets[i] | sets[j])
            vals.append(inter / uni if uni > 0 else 1.0)
    return float(np.nanmedian(vals)) if vals else float('nan')


def run_pisa_sensitivity(out_dir: str,
                         linkages: List[str],
                         metrics: List[str],
                         noises: List[float],
                         seeds: List[int],
                         n_clusters: int = 4,
                         export_excel: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    df_original = load_pisa()

    qual = ['Country', 'Type of Economy', 'Population Density', 'Type of Government', 'Continent']
    df_num0 = df_original.drop(columns=qual)

    summary_rows = []
    ari_labels_by_config: dict[tuple, list[np.ndarray]] = {}

    for linkage in linkages:
        for metric in metrics:
            if linkage == 'ward' and metric != 'euclidean':
                # Ward is defined with Euclidean distances on observations
                continue
            for noise in noises:
                for seed in seeds:
                    rng = np.random.RandomState(seed)
                    df_num = add_noise(df_num0, noise, rng)
                    df_scaled = stand(df_num)

                    Base = build_base(df_scaled, linkage_method=linkage, distance_metric=metric)
                    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=metric)
                    gen_base_2 = build_gen_base_2(Base, df_scaled, df_num, distance_metric=metric)
                    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=seed)
                    # Store labels for ARI stability across seeds
                    key = ("PISA", linkage, metric, noise)
                    labels = _labels_from_groups(clusters, n=len(df_num))
                    ari_labels_by_config.setdefault(key, []).append(labels)

                    (A, target_groups, g_econ, g_dens, g_gov, g_cont) = pisa_define_groups(
                        df_original, gen_base_1, gen_base_2, Base, clusters
                    )

                    families = {
                        "economy": g_econ,
                        "density": g_dens,
                        "government": g_gov,
                        "continent": g_cont,
                    }

                    rows = summarize_family(df_original, A, families, target_groups)
                    for r in rows:
                        r.update({
                            "dataset": "PISA",
                            "linkage": linkage,
                            "metric": metric,
                            "noise": noise,
                            "random_state": seed,
                        })
                    summary_rows.extend(rows)

                    # Optional: export Excel for this config
                    if export_excel:
                        from utils.analysis import process_batch
                        base_out = os.path.join(out_dir, f"PISA_l-{linkage}_m-{metric}_n-{noise}_s-{seed}")
                        for fam_name, groups in families.items():
                            fam_dir = os.path.join(base_out, fam_name)
                            os.makedirs(fam_dir, exist_ok=True)
                            process_batch(groups, df_original, target_groups, A, output_folder=fam_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Compute ARI stability across seeds per (dataset, linkage, metric, noise)
    ari_rows = []
    for key, label_list in ari_labels_by_config.items():
        if len(label_list) < 2:
            continue
        pairwise = []
        for i in range(len(label_list)):
            for j in range(i+1, len(label_list)):
                pairwise.append(adjusted_rand_score(label_list[i], label_list[j]))
        ari_rows.append({
            "dataset": key[0],
            "linkage": key[1],
            "metric": key[2],
            "noise": key[3],
            "ari_median": float(np.median(pairwise)),
            "ari_mean": float(np.mean(pairwise)),
            "pairs": len(pairwise),
        })
    pd.DataFrame(ari_rows).to_csv(os.path.join(out_dir, "stability_ari.csv"), index=False)

    # Compute Jaccard stability of closures across seeds per (dataset, linkage, metric, noise, family, seed, base)
    jacc_rows = []
    if not summary_df.empty:
        # Group by key, then aggregate Jaccard over random_state
        group_cols = ["dataset", "linkage", "metric", "noise", "family", "seed", "base"]
        for key, df_sub in summary_df.groupby(group_cols):
            # For each random_state, collect closure set
            sets = []
            for _, row in df_sub.iterrows():
                s = set(int(i) for i in str(row["closure_indices"]).split(',') if i != '')
                sets.append(s)
            jac = _pairwise_jaccard(sets)
            jacc_rows.append({
                "dataset": key[0],
                "linkage": key[1],
                "metric": key[2],
                "noise": key[3],
                "family": key[4],
                "seed": key[5],
                "base": key[6],
                "jaccard_median": jac,
                "runs": len(sets),
            })
    pd.DataFrame(jacc_rows).to_csv(os.path.join(out_dir, "stability_jaccard.csv"), index=False)


def run_gapminder_sensitivity(out_dir: str,
                              linkages: List[str],
                              metrics: List[str],
                              noises: List[float],
                              seeds: List[int],
                              n_clusters: int = 4,
                              export_excel: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    df_all0 = prepare_gapminder_latest()

    # Numeric selection (mirror the runner)
    numeric_df0 = df_all0.select_dtypes(include=[np.number]).copy()
    drop_ids = [c for c in ['year', 'iso_num'] if c in numeric_df0.columns]
    df_num0 = numeric_df0.drop(columns=drop_ids)

    summary_rows = []
    ari_labels_by_config: dict[tuple, list[np.ndarray]] = {}

    for linkage in linkages:
        for metric in metrics:
            if linkage == 'ward' and metric != 'euclidean':
                continue
            for noise in noises:
                for seed in seeds:
                    rng = np.random.RandomState(seed)
                    df_num = add_noise(df_num0, noise, rng)
                    df_scaled = stand(df_num)

                    Base = build_base(df_scaled, linkage_method=linkage, distance_metric=metric)
                    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=metric)
                    gen_base_2 = build_gen_base_2(Base, df_scaled, df_num, distance_metric=metric)
                    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=seed)
                    # Store labels for ARI
                    key = ("Gapminder", linkage, metric, noise)
                    labels = _labels_from_groups(clusters, n=len(df_num))
                    ari_labels_by_config.setdefault(key, []).append(labels)

                    A = [str(i) for i in range(len(df_all0))]
                    target_groups = {
                        'gen_base_1': gen_base_1,
                        'gen_base_2': gen_base_2,
                        'base': Base,
                        'clusters': clusters,
                    }
                    families = define_groups_gapminder(df_all0)

                    rows = summarize_family(df_all0.rename(columns={'country': 'Country'}), A, families, target_groups)
                    for r in rows:
                        r.update({
                            "dataset": "Gapminder",
                            "linkage": linkage,
                            "metric": metric,
                            "noise": noise,
                            "random_state": seed,
                        })
                    summary_rows.extend(rows)

                    if export_excel:
                        from utils.analysis import process_batch
                        base_out = os.path.join(out_dir, f"Gapminder_l-{linkage}_m-{metric}_n-{noise}_s-{seed}")
                        for fam_name, groups in families.items():
                            fam_dir = os.path.join(base_out, fam_name)
                            os.makedirs(fam_dir, exist_ok=True)
                            process_batch(groups, df_all0.rename(columns={'country':'Country'}), target_groups, A, output_folder=fam_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ARI stability
    ari_rows = []
    for key, label_list in ari_labels_by_config.items():
        if len(label_list) < 2:
            continue
        pairwise = []
        for i in range(len(label_list)):
            for j in range(i+1, len(label_list)):
                pairwise.append(adjusted_rand_score(label_list[i], label_list[j]))
        ari_rows.append({
            "dataset": key[0],
            "linkage": key[1],
            "metric": key[2],
            "noise": key[3],
            "ari_median": float(np.median(pairwise)),
            "ari_mean": float(np.mean(pairwise)),
            "pairs": len(pairwise),
        })
    pd.DataFrame(ari_rows).to_csv(os.path.join(out_dir, "stability_ari.csv"), index=False)

    # Jaccard stability of closures
    jacc_rows = []
    if not summary_df.empty:
        group_cols = ["dataset", "linkage", "metric", "noise", "family", "seed", "base"]
        for key, df_sub in summary_df.groupby(group_cols):
            sets = []
            for _, row in df_sub.iterrows():
                s = set(int(i) for i in str(row["closure_indices"]).split(',') if i != '')
                sets.append(s)
            jac = _pairwise_jaccard(sets)
            jacc_rows.append({
                "dataset": key[0],
                "linkage": key[1],
                "metric": key[2],
                "noise": key[3],
                "family": key[4],
                "seed": key[5],
                "base": key[6],
                "jaccard_median": jac,
                "runs": len(sets),
            })
    pd.DataFrame(jacc_rows).to_csv(os.path.join(out_dir, "stability_jaccard.csv"), index=False)


def run_uci_sensitivity(out_dir: str,
                        dataset_name: str,
                        linkages: List[str],
                        metrics: List[str],
                        noises: List[float],
                        seeds: List[int],
                        n_clusters: int = 4,
                        export_excel: bool = False):
    """Sensitivity analysis for UCI datasets (iris, wine, breast_cancer)."""
    os.makedirs(out_dir, exist_ok=True)
    X0, y0 = load_uci(dataset_name)
    # Subsample to at most 100 points, roughly balanced across classes
    max_samples = 100
    if len(X0) > max_samples:
        sample_seed = seeds[0] if seeds else 42
        idx = _balanced_subsample_indices_array(y0.to_numpy(), max_samples, random_state=sample_seed)
        X0 = X0.iloc[idx].reset_index(drop=True)
        y0 = y0.iloc[idx].reset_index(drop=True)
    df_num0 = X0.copy()

    dataset_label = f"UCI-{dataset_name}"

    summary_rows = []
    ari_labels_by_config: dict[tuple, list[np.ndarray]] = {}

    for linkage in linkages:
        for metric in metrics:
            if linkage == "ward" and metric != "euclidean":
                continue
            for noise in noises:
                for seed in seeds:
                    rng = np.random.RandomState(seed)
                    df_num = add_noise(df_num0, noise, rng)
                    df_all = df_num.copy()
                    df_all["label"] = y0.values
                    df_all["Country"] = [f"P{i}" for i in range(len(df_all))]

                    df_scaled = stand(df_num)

                    Base = build_base(df_scaled, linkage_method=linkage, distance_metric=metric)
                    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=metric)
                    gen_base_2 = build_gen_base_2(Base, df_scaled, df_num, distance_metric=metric)
                    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=seed)

                    key = (dataset_label, linkage, metric, noise)
                    labels = _labels_from_groups(clusters, n=len(df_num))
                    ari_labels_by_config.setdefault(key, []).append(labels)

                    A = [str(i) for i in range(len(df_all))]
                    target_groups = {
                        "gen_base_1": gen_base_1,
                        "gen_base_2": gen_base_2,
                        "base": Base,
                        "clusters": clusters,
                    }
                    families = define_groups_uci(df_all)

                    from utils.analysis import process_batch

                    rows = summarize_family(df_all, A, families, target_groups)
                    for r in rows:
                        fam = r["family"]
                        seed_name = r["seed"]
                        # default semantic metrics
                        closure_purity = np.nan
                        interior_purity = np.nan
                        label_family = None
                        label_value = None

                        if fam == "label":
                            label_family = "label"
                            try:
                                label_value = int(str(seed_name).split("_")[-1])
                            except Exception:
                                label_value = None
                            if label_value is not None:
                                label_col = "label"
                                clos_idx = [int(i) for i in str(r.get("closure_indices", "")).split(",") if i != ""]
                                inte_idx = [int(i) for i in str(r.get("interior_indices", "")).split(",") if i != ""]
                                if clos_idx:
                                    labels_clos = df_all[label_col].iloc[clos_idx].to_numpy()
                                    closure_purity = float((labels_clos == label_value).mean())
                                if inte_idx:
                                    labels_inte = df_all[label_col].iloc[inte_idx].to_numpy()
                                    interior_purity = float((labels_inte == label_value).mean())

                        r.update(
                            {
                                "dataset": dataset_label,
                                "linkage": linkage,
                                "metric": metric,
                                "noise": noise,
                                "random_state": seed,
                                "label_family": label_family,
                                "label_value": label_value,
                                "closure_purity": closure_purity,
                                "interior_purity": interior_purity,
                            }
                        )
                    summary_rows.extend(rows)

                    if export_excel:
                        base_out = os.path.join(out_dir, f"{dataset_label}_l-{linkage}_m-{metric}_n-{noise}_s-{seed}")
                        for fam_name, groups in families.items():
                            fam_dir = os.path.join(base_out, fam_name)
                            os.makedirs(fam_dir, exist_ok=True)
                            process_batch(groups, df_all, target_groups, A, output_folder=fam_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ARI stability
    ari_rows = []
    for key, label_list in ari_labels_by_config.items():
        if len(label_list) < 2:
            continue
        pairwise = []
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                pairwise.append(adjusted_rand_score(label_list[i], label_list[j]))
        ari_rows.append(
            {
                "dataset": key[0],
                "linkage": key[1],
                "metric": key[2],
                "noise": key[3],
                "ari_median": float(np.median(pairwise)),
                "ari_mean": float(np.mean(pairwise)),
                "pairs": len(pairwise),
            }
        )
    pd.DataFrame(ari_rows).to_csv(os.path.join(out_dir, "stability_ari.csv"), index=False)

    # Jaccard stability of closures
    jacc_rows = []
    if not summary_df.empty:
        group_cols = ["dataset", "linkage", "metric", "noise", "family", "seed", "base"]
        for key, df_sub in summary_df.groupby(group_cols):
            sets = []
            for _, row in df_sub.iterrows():
                s = set(int(i) for i in str(row["closure_indices"]).split(",") if i != "")
                sets.append(s)
            jac = _pairwise_jaccard(sets)
            jacc_rows.append(
                {
                    "dataset": key[0],
                    "linkage": key[1],
                    "metric": key[2],
                    "noise": key[3],
                    "family": key[4],
                    "seed": key[5],
                    "base": key[6],
                    "jaccard_median": jac,
                    "runs": len(sets),
                }
            )
    pd.DataFrame(jacc_rows).to_csv(os.path.join(out_dir, "stability_jaccard.csv"), index=False)


def run_synthetic_sensitivity(out_dir: str,
                              kind: str,
                              n_samples: int,
                              linkages: List[str],
                              metrics: List[str],
                              noises: List[float],
                              seeds: List[int],
                              n_clusters: int = 4,
                              export_excel: bool = False):
    """Sensitivity analysis for synthetic datasets (blobs/moons/circles).

    Uses generator noise and seeds to vary data; evaluates bases against
    cluster_true (ground truth) and regime (orthogonal label).
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset_name = f"Synthetic-{kind}"

    summary_rows = []
    ari_labels_by_config: dict[tuple, list[np.ndarray]] = {}

    for linkage in linkages:
        for metric in metrics:
            if linkage == "ward" and metric != "euclidean":
                continue
            for noise in noises:
                for seed in seeds:
                    # Generate synthetic data with given noise/seed
                    X, y_cluster = make_synthetic(kind=kind, n_samples=n_samples, noise=noise, random_state=seed)
                    df_all = X.copy()
                    df_all["cluster_true"] = y_cluster.astype(int)

                    # Define orthogonal regime label via random linear rule
                    rng = np.random.RandomState(seed)
                    w = rng.normal(size=X.shape[1])
                    scores = X.values @ w
                    threshold = np.median(scores)
                    df_all["regime"] = (scores >= threshold).astype(int)

                    # Synthetic identifier
                    df_all["Country"] = [f"P{i}" for i in range(len(df_all))]

                    # Numeric features only for bases/clusters
                    df_num = df_all.drop(columns=["cluster_true", "regime", "Country"])
                    df_scaled = stand(df_num)

                    Base = build_base(df_scaled, linkage_method=linkage, distance_metric=metric)
                    gen_base_1 = build_gen_base_1(Base, df_num, distance_metric=metric)
                    gen_base_2 = build_gen_base_2(Base, df_scaled, df_num, distance_metric=metric)
                    clusters = kmeans_clusters_from_columns(df_num, df_num.columns, n_clusters=n_clusters, random_state=seed)

                    # Store labels for ARI across seeds
                    key = (dataset_name, linkage, metric, noise)
                    labels = _labels_from_groups(clusters, n=len(df_num))
                    ari_labels_by_config.setdefault(key, []).append(labels)

                    A = [str(i) for i in range(len(df_all))]
                    target_groups = {
                        "gen_base_1": gen_base_1,
                        "gen_base_2": gen_base_2,
                        "base": Base,
                        "clusters": clusters,
                    }
                    families = define_groups_synthetic(df_all)

                    from utils.analysis import process_batch

                    rows = summarize_family(df_all, A, families, target_groups)
                    for r in rows:
                        fam = r["family"]
                        seed_name = r["seed"]
                        # Default label metrics as NaN
                        closure_purity = np.nan
                        interior_purity = np.nan
                        label_family = None
                        label_value = None

                        if fam in ("cluster_true", "regime"):
                            label_family = fam
                            try:
                                label_value = int(str(seed_name).split("_")[-1])
                            except Exception:
                                label_value = None
                            if label_value is not None:
                                label_col = fam
                                # Parse closure/interior indices
                                clos_idx = [int(i) for i in str(r.get("closure_indices", "")).split(",") if i != ""]
                                inte_idx = [int(i) for i in str(r.get("interior_indices", "")).split(",") if i != ""]
                                if clos_idx:
                                    labels_clos = df_all[label_col].iloc[clos_idx].to_numpy()
                                    closure_purity = float((labels_clos == label_value).mean())
                                if inte_idx:
                                    labels_inte = df_all[label_col].iloc[inte_idx].to_numpy()
                                    interior_purity = float((labels_inte == label_value).mean())

                        r.update(
                            {
                                "dataset": dataset_name,
                                "linkage": linkage,
                                "metric": metric,
                                "noise": noise,
                                "random_state": seed,
                                "label_family": label_family,
                                "label_value": label_value,
                                "closure_purity": closure_purity,
                                "interior_purity": interior_purity,
                            }
                        )
                    summary_rows.extend(rows)

                    if export_excel:
                        base_out = os.path.join(out_dir, f"{dataset_name}_l-{linkage}_m-{metric}_n-{noise}_s-{seed}")
                        for fam_name, groups in families.items():
                            fam_dir = os.path.join(base_out, fam_name)
                            os.makedirs(fam_dir, exist_ok=True)
                            process_batch(groups, df_all, target_groups, A, output_folder=fam_dir)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ARI stability
    ari_rows = []
    for key, label_list in ari_labels_by_config.items():
        if len(label_list) < 2:
            continue
        pairwise = []
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                pairwise.append(adjusted_rand_score(label_list[i], label_list[j]))
        ari_rows.append(
            {
                "dataset": key[0],
                "linkage": key[1],
                "metric": key[2],
                "noise": key[3],
                "ari_median": float(np.median(pairwise)),
                "ari_mean": float(np.mean(pairwise)),
                "pairs": len(pairwise),
            }
        )
    pd.DataFrame(ari_rows).to_csv(os.path.join(out_dir, "stability_ari.csv"), index=False)

    # Jaccard stability of closures
    jacc_rows = []
    if not summary_df.empty:
        group_cols = ["dataset", "linkage", "metric", "noise", "family", "seed", "base"]
        for key, df_sub in summary_df.groupby(group_cols):
            sets = []
            for _, row in df_sub.iterrows():
                s = set(int(i) for i in str(row["closure_indices"]).split(",") if i != "")
                sets.append(s)
            jac = _pairwise_jaccard(sets)
            jacc_rows.append(
                {
                    "dataset": key[0],
                    "linkage": key[1],
                    "metric": key[2],
                    "noise": key[3],
                    "family": key[4],
                    "seed": key[5],
                    "base": key[6],
                    "jaccard_median": jac,
                    "runs": len(sets),
                }
            )
    pd.DataFrame(jacc_rows).to_csv(os.path.join(out_dir, "stability_jaccard.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sensitivity runner for PISA, Gapminder, Synthetic, and UCI data.")
    parser.add_argument('--dataset', default='pisa', choices=['pisa', 'gapminder', 'synthetic', 'uci'], help='Dataset to analyze')
    parser.add_argument('--out', default=None, help='Output directory (defaults to results/<DATASET>/sensitivity)')
    parser.add_argument('--linkages', default='single,complete,ward', help='Comma-separated linkage methods')
    parser.add_argument('--metrics', default='euclidean,cityblock', help='Comma-separated distance metrics (used unless linkage=ward)')
    parser.add_argument('--noises', default='0.0,0.02', help='Comma-separated noise levels')
    parser.add_argument('--seeds', default='11,23,37,51,73', help='Comma-separated random seeds (for KMeans and noise/generation)')
    parser.add_argument('--n-clusters', type=int, default=4, help='Number of clusters for k-means base')
    parser.add_argument('--export-excel', action='store_true', help='Also export per-run Excel outputs (heavier)')
    # Synthetic-specific options
    parser.add_argument('--synthetic-kind', default='blobs', choices=['blobs', 'moons', 'circles'], help='Synthetic dataset kind')
    parser.add_argument('--synthetic-n-samples', type=int, default=80, help='Number of synthetic samples per run')
    # UCI-specific options
    parser.add_argument('--uci-name', default='iris', choices=['iris', 'wine', 'breast_cancer'], help='UCI dataset name')

    args = parser.parse_args()

    linkages = [s.strip() for s in args.linkages.split(',') if s.strip()]
    metrics = [s.strip() for s in args.metrics.split(',') if s.strip()]
    noises = [float(s.strip()) for s in args.noises.split(',') if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    if args.out:
        out_dir = args.out
    else:
        if args.dataset == 'pisa':
            root = 'PISA'
        elif args.dataset == 'gapminder':
            root = 'Gapminder'
        elif args.dataset == 'synthetic':
            root = 'Synthetic'
        else:
            root = f"UCI-{args.uci_name}"
        out_dir = os.path.join(_ROOT, 'results', root, 'sensitivity')

    if args.dataset == 'pisa':
        run_pisa_sensitivity(out_dir, linkages, metrics, noises, seeds, n_clusters=args.n_clusters, export_excel=args.export_excel)
    elif args.dataset == 'gapminder':
        run_gapminder_sensitivity(out_dir, linkages, metrics, noises, seeds, n_clusters=args.n_clusters, export_excel=args.export_excel)
    elif args.dataset == 'synthetic':
        run_synthetic_sensitivity(
            out_dir,
            kind=args.synthetic_kind,
            n_samples=args.synthetic_n_samples,
            linkages=linkages,
            metrics=metrics,
            noises=noises,
            seeds=seeds,
            n_clusters=args.n_clusters,
            export_excel=args.export_excel,
        )
    else:
        run_uci_sensitivity(
            out_dir,
            dataset_name=args.uci_name,
            linkages=linkages,
            metrics=metrics,
            noises=noises,
            seeds=seeds,
            n_clusters=args.n_clusters,
            export_excel=args.export_excel,
        )
