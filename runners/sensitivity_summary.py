import os
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path shim for repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    required = {"dataset", "family", "seed", "base", "closure_size", "interior_size", "seed_size", "n_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary: {missing}")
    return df


def median_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Median closure/interior per dataset/family/seed/base across runs
    value_cols = ["closure_size", "interior_size", "seed_size", "n_total"]
    purity_cols = []
    if "closure_purity" in df.columns and "interior_purity" in df.columns:
        purity_cols = ["closure_purity", "interior_purity"]
        value_cols += purity_cols

    med = (
        df.groupby(["dataset", "family", "seed", "base"])[value_cols]
          .median()
          .reset_index()
    )
    # Derived ratios
    med["expansion_ratio"] = med["closure_size"] / med["seed_size"].replace(0, pd.NA)
    med["retention_ratio"] = med["interior_size"] / med["seed_size"].replace(0, pd.NA)
    med["closure_prop"] = med["closure_size"] / med["n_total"].replace(0, pd.NA)

    agg_cols = ["closure_size", "interior_size", "expansion_ratio", "retention_ratio", "closure_prop"]
    if purity_cols:
        agg_cols += purity_cols

    # Aggregate further across seeds within family to get per-base tendency
    med_by_base = (
        med.groupby(["dataset", "family", "base"])[agg_cols]
           .median()
           .reset_index()
    )
    return med, med_by_base


def plot_medians(med_by_base: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for dataset in med_by_base["dataset"].unique():
        metrics = ["closure_size", "interior_size", "expansion_ratio", "retention_ratio", "closure_purity", "interior_purity"]
        for metric in metrics:
            if metric not in med_by_base.columns:
                continue
            dfp = med_by_base[med_by_base["dataset"] == dataset].copy()
            plt.figure(figsize=(10, 5))
            sns.barplot(dfp, x="family", y=metric, hue="base")
            plt.title(f"{dataset} median {metric} by base and family")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{dataset}_median_{metric}.png")
            plt.savefig(fname)
            plt.close()


def plot_ari(ari_csv: str, out_dir: str) -> None:
    if not os.path.exists(ari_csv):
        return
    df = pd.read_csv(ari_csv)
    if df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    for dataset in df["dataset"].unique():
        dfd = df[df["dataset"] == dataset]
        for linkage in sorted(dfd["linkage"].unique()):
            dfl = dfd[dfd["linkage"] == linkage]
            plt.figure(figsize=(10, 5))
            sns.lineplot(dfl, x="noise", y="ari_median", hue="metric", marker="o")
            plt.ylim(0, 1)
            plt.title(f"{dataset} ARI median across seeds (linkage={linkage})")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{dataset}_ari_linkage-{linkage}.png")
            plt.savefig(fname)
            plt.close()


def plot_jaccard(jacc_csv: str, out_dir: str) -> None:
    if not os.path.exists(jacc_csv):
        return
    df = pd.read_csv(jacc_csv)
    if df.empty:
        return
    # Aggregate across seed groups to summarize per family/base
    agg = (
        df.groupby(["dataset", "family", "base", "linkage", "metric", "noise"]) ["jaccard_median"]
          .median()
          .reset_index()
    )
    os.makedirs(out_dir, exist_ok=True)
    for dataset in agg["dataset"].unique():
        dfd = agg[agg["dataset"] == dataset]
        for fam in dfd["family"].unique():
            dff = dfd[dfd["family"] == fam]
            for linkage in sorted(dff["linkage"].unique()):
                dfl = dff[dff["linkage"] == linkage]
                plt.figure(figsize=(10, 5))
                sns.lineplot(dfl, x="noise", y="jaccard_median", hue="base", style="metric", marker="o")
                plt.ylim(0, 1)
                plt.title(f"{dataset} {fam} closure Jaccard (linkage={linkage})")
                plt.tight_layout()
                fname = os.path.join(out_dir, f"{dataset}_{fam}_jaccard_linkage-{linkage}.png")
                plt.savefig(fname)
                plt.close()


def highlight_synthetic(med_by_base: pd.DataFrame) -> None:
    """Print a short textual summary of synthetic results based on purity."""
    if "closure_purity" not in med_by_base.columns or "interior_purity" not in med_by_base.columns:
        return

    synth = med_by_base[med_by_base["dataset"].str.startswith("Synthetic")].copy()
    if synth.empty:
        return

    print("\n=== Synthetic dataset highlights ===")
    for dataset in sorted(synth["dataset"].unique()):
        df_d = synth[synth["dataset"] == dataset]
        print(f"\nDataset: {dataset}")
        for fam in sorted(df_d["family"].unique()):
            df_f = df_d[df_d["family"] == fam]
            if df_f.empty:
                continue
            # Best bases by closure/interior purity
            best_closure = df_f.loc[df_f["closure_purity"].idxmax()]
            best_interior = df_f.loc[df_f["interior_purity"].idxmax()]

            # Extract reference rows for key bases if present
            def row_for(base_name: str):
                rows = df_f[df_f["base"] == base_name]
                return rows.iloc[0] if not rows.empty else None

            row_g1 = row_for("gen_base_1")
            row_g2 = row_for("gen_base_2")
            row_clusters = row_for("clusters")

            print(f"  Family: {fam}")
            print(f"    Highest closure purity: {best_closure['base']} = {best_closure['closure_purity']:.3f}")
            print(f"    Highest interior purity: {best_interior['base']} = {best_interior['interior_purity']:.3f}")

            # If we have generalized vs clusters, compare them explicitly
            if row_clusters is not None and (row_g1 is not None or row_g2 is not None):
                msgs = []
                if row_g1 is not None:
                    msgs.append(
                        f"gen_base_1 vs clusters (closure_purity): {row_g1['closure_purity']:.3f} vs {row_clusters['closure_purity']:.3f}"
                    )
                if row_g2 is not None:
                    msgs.append(
                        f"gen_base_2 vs clusters (closure_purity): {row_g2['closure_purity']:.3f} vs {row_clusters['closure_purity']:.3f}"
                    )
                for m in msgs:
                    print(f"    {m}")


def _highlight_dataset_family(med_by_base: pd.DataFrame, dataset_filter, title: str, use_purity: bool = False) -> None:
    df = med_by_base[dataset_filter(med_by_base["dataset"])].copy()
    if df.empty:
        return
    print(f"\n=== {title} ===")
    for dataset in sorted(df["dataset"].unique()):
        df_d = df[df["dataset"] == dataset]
        print(f"\nDataset: {dataset}")
        for fam in sorted(df_d["family"].unique()):
            df_f = df_d[df_d["family"] == fam]
            if df_f.empty:
                continue
            print(f"  Family: {fam}")
            if use_purity and "closure_purity" in df_f.columns and "interior_purity" in df_f.columns:
                # Only use purity where it is actually defined
                mask_cl = df_f["closure_purity"].notna()
                mask_in = df_f["interior_purity"].notna()
                if mask_cl.any() and mask_in.any():
                    best_closure = df_f.loc[df_f.loc[mask_cl, "closure_purity"].idxmax()]
                    best_interior = df_f.loc[df_f.loc[mask_in, "interior_purity"].idxmax()]
                    print(f"    Highest closure purity: {best_closure['base']} = {best_closure['closure_purity']:.3f}")
                    print(f"    Highest interior purity: {best_interior['base']} = {best_interior['interior_purity']:.3f}")
                    continue  # Skip to next family since purity summary is available

            # Fallback: highlight via expansion/retention ratios
            best_retention = df_f.loc[df_f["retention_ratio"].idxmax()]
            best_expansion = df_f.loc[df_f["expansion_ratio"].idxmax()]
            print(f"    Highest retention_ratio: {best_retention['base']} = {best_retention['retention_ratio']:.3f}")
            print(f"    Highest expansion_ratio: {best_expansion['base']} = {best_expansion['expansion_ratio']:.3f}")


def highlight_uci(med_by_base: pd.DataFrame) -> None:
    """Highlight UCI datasets based on label purity, if available."""
    def filt(series):
        return series.str.startswith("UCI-")
    _highlight_dataset_family(med_by_base, filt, "UCI dataset highlights", use_purity=True)


def highlight_gapminder_pisa(med_by_base: pd.DataFrame) -> None:
    """Highlight Gapminder and PISA with expansion/retention summaries."""
    def filt(series):
        return series.isin(["PISA", "Gapminder"])
    _highlight_dataset_family(med_by_base, filt, "PISA/Gapminder highlights", use_purity=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize sensitivity results and plot medians.")
    parser.add_argument("--summary", required=True, help="Path to summary.csv from sensitivity runner")
    parser.add_argument("--out", default=None, help="Output directory for tables and plots (default: alongside summary)")
    parser.add_argument("--ari", default=None, help="Path to stability_ari.csv (optional, defaults beside summary)")
    parser.add_argument("--jacc", default=None, help="Path to stability_jaccard.csv (optional, defaults beside summary)")
    args = parser.parse_args()

    # Normalize summary path to repo root when given as relative 'results/...'
    summary_path = args.summary
    if not os.path.isabs(summary_path) and summary_path.startswith("results"):
        summary_path = os.path.join(_ROOT, summary_path)

    df = load_summary(summary_path)
    med, med_by_base = median_tables(df)

    out_root = args.out or os.path.join(os.path.dirname(summary_path), "summary")
    os.makedirs(out_root, exist_ok=True)

    med.to_csv(os.path.join(out_root, "medians_per_seed.csv"), index=False)
    med_by_base.to_csv(os.path.join(out_root, "medians_by_base.csv"), index=False)

    plots_dir = os.path.join(out_root, "plots")
    plot_medians(med_by_base, plots_dir)
    # Stability plots
    ari_csv = args.ari or os.path.join(os.path.dirname(summary_path), "stability_ari.csv")
    jacc_csv = args.jacc or os.path.join(os.path.dirname(summary_path), "stability_jaccard.csv")
    plot_ari(ari_csv, plots_dir)
    plot_jaccard(jacc_csv, plots_dir)

    # Print synthetic highlights, if applicable
    highlight_synthetic(med_by_base)
    highlight_uci(med_by_base)
    highlight_gapminder_pisa(med_by_base)


if __name__ == "__main__":
    main()
