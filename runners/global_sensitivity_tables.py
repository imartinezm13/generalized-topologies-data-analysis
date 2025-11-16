import os
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "results"

# Folder roots under results/
# For Synthetic, this script expects results/Synthetic/sensitivity/...
# For UCI, use the same root names as in sensitivity_runner (e.g. UCI-iris).
DATASET_ROOTS: List[str] = [
    "PISA",
    "Gapminder",
    "Synthetic",
    "UCI-iris",
]


def _load_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def load_medians_by_base(dataset_root: str) -> pd.DataFrame:
    path = ROOT / dataset_root / "sensitivity" / "summary" / "medians_by_base.csv"
    return _load_csv_or_empty(path)


def load_stability_jaccard(dataset_root: str) -> pd.DataFrame:
    path = ROOT / dataset_root / "sensitivity" / "stability_jaccard.csv"
    return _load_csv_or_empty(path)


def load_stability_ari(dataset_root: str) -> pd.DataFrame:
    path = ROOT / dataset_root / "sensitivity" / "stability_ari.csv"
    return _load_csv_or_empty(path)


def build_quality_table() -> pd.DataFrame:
    """Aggregate medians_by_base across datasets into one wide table."""
    rows = []
    for root in DATASET_ROOTS:
        df = load_medians_by_base(root)
        if df.empty:
            continue
        # One row per (dataset, base), aggregating across families
        agg = (
            df.groupby(["dataset", "base"])
              .median(numeric_only=True)
              .reset_index()
        )
        for _, r in agg.iterrows():
            rows.append(
                {
                    "dataset": r["dataset"],
                    "base": r["base"],
                    "expansion_ratio_med": r.get("expansion_ratio", float("nan")),
                    "retention_ratio_med": r.get("retention_ratio", float("nan")),
                    "closure_purity_med": r.get("closure_purity", float("nan")),
                    "interior_purity_med": r.get("interior_purity", float("nan")),
                }
            )
    qual = pd.DataFrame(rows)
    return qual


def build_stability_table() -> pd.DataFrame:
    """Aggregate Jaccard and ARI stability into a compact table."""
    rows = []
    for root in DATASET_ROOTS:
        jdf = load_stability_jaccard(root)
        if not jdf.empty:
            j_agg = (
                jdf.groupby(["dataset", "base"])
                   .agg(jaccard_med=("jaccard_median", "median"))
                   .reset_index()
            )
        else:
            j_agg = pd.DataFrame(columns=["dataset", "base", "jaccard_med"])

        adf = load_stability_ari(root)
        if not adf.empty:
            a_agg = (
                adf.groupby("dataset")
                   .agg(ari_med=("ari_median", "median"))
                   .reset_index()
            )
        else:
            a_agg = pd.DataFrame(columns=["dataset", "ari_med"])

        # Merge Jaccard + ARI into per-dataset/base rows
        for base in ["base", "gen_base_1", "gen_base_2", "clusters"]:
            jd = j_agg[j_agg["base"] == base]
            for dataset in jd["dataset"].unique():
                jmed = float(jd[jd["dataset"] == dataset]["jaccard_med"].median())
                if base == "clusters":
                    amed_series = a_agg[a_agg["dataset"] == dataset]["ari_med"]
                    amed = float(amed_series.median()) if not amed_series.empty else float("nan")
                else:
                    amed = float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "base": base,
                        "jaccard_med": jmed,
                        "ari_med": amed,
                    }
                )

    stab = pd.DataFrame(rows)
    return stab


def to_latex_tables() -> None:
    qual = build_quality_table()
    stab = build_stability_table()

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    if not qual.empty:
        # Example: expansion ratios by base/dataset
        exp_pivot = qual.pivot(index="base", columns="dataset", values="expansion_ratio_med")
        print("% Expansion ratios by base/dataset")
        print(exp_pivot.to_latex(float_format="%.3f"))
        print()

        # Optionally, retention or purity can be pivoted similarly
        ret_pivot = qual.pivot(index="base", columns="dataset", values="retention_ratio_med")
        print("% Retention ratios by base/dataset")
        print(ret_pivot.to_latex(float_format="%.3f"))
        print()

        if "closure_purity_med" in qual.columns and not qual["closure_purity_med"].isna().all():
            pur_pivot = qual.pivot(index="base", columns="dataset", values="closure_purity_med")
            print("% Closure purity by base/dataset (when available)")
            print(pur_pivot.to_latex(float_format="%.3f"))
            print()

    if not stab.empty:
        print("% Stability table (median Jaccard and ARI)")
        print(stab.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    to_latex_tables()

