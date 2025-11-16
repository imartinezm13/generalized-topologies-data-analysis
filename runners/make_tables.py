import os
import sys
import re
from typing import List

import pandas as pd

# Ensure repository root is on sys.path when running directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_summary(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    required = {"dataset", "family", "seed", "base", "closure_size", "interior_size", "seed_size", "n_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary: {missing}")
    return df


def _compute_medians(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["closure_size", "interior_size", "seed_size", "n_total"]
    purity_cols: List[str] = []
    if "closure_purity" in df.columns and "interior_purity" in df.columns:
        purity_cols = ["closure_purity", "interior_purity"]
        value_cols += purity_cols

    med = (
        df.groupby(["dataset", "family", "seed", "base"])[value_cols]
          .median()
          .reset_index()
    )
    med["expansion_ratio"] = med["closure_size"] / med["seed_size"].replace(0, pd.NA)
    med["retention_ratio"] = med["interior_size"] / med["seed_size"].replace(0, pd.NA)
    med["closure_prop"] = med["closure_size"] / med["n_total"].replace(0, pd.NA)

    agg_cols = ["closure_size", "interior_size", "expansion_ratio", "retention_ratio", "closure_prop"]
    if purity_cols:
        agg_cols += purity_cols

    med_by_base = (
        med.groupby(["dataset", "family", "base"])[agg_cols]
           .median()
           .reset_index()
    )
    return med_by_base


def _sanitize_label(text: str) -> str:
    """Sanitize a string for use in LaTeX labels."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _format_float(v) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "-"


def make_tables(summary_path: str, out_path: str) -> None:
    df = _load_summary(summary_path)
    med_by_base = _compute_medians(df)

    lines: List[str] = []
    lines.append("% Auto-generated tables from sensitivity summary")

    for dataset in med_by_base["dataset"].unique():
        df_d = med_by_base[med_by_base["dataset"] == dataset]
        for family in df_d["family"].unique():
            df_f = df_d[df_d["family"] == family].copy()
            if df_f.empty:
                continue

            include_purity = "closure_purity" in df_f.columns and not df_f["closure_purity"].isna().all()

            cols = ["closure_size", "interior_size", "expansion_ratio", "retention_ratio"]
            if include_purity:
                cols += ["closure_purity", "interior_purity"]

            col_titles = {
                "closure_size": "Closure size",
                "interior_size": "Interior size",
                "expansion_ratio": "Expansion",
                "retention_ratio": "Retention",
                "closure_purity": "Closure purity",
                "interior_purity": "Interior purity",
            }

            ncols = 1 + len(cols)
            align = "l" + "r" * len(cols)

            label = f"tab:{_sanitize_label(dataset)}-{_sanitize_label(family)}"
            caption = f"Median closure/interior metrics for dataset {dataset}, family {family}."

            lines.append("")
            lines.append("\\begin{table}[ht]")
            lines.append("  \\centering")
            # Escape braces around 'tabular' for f-string
            lines.append(f"  \\begin{{tabular}}{{{align}}}")
            lines.append("    \\hline")
            header_cols = ["Base"] + [col_titles[c] for c in cols]
            lines.append("    " + " & ".join(header_cols) + " \\\\")
            lines.append("    \\hline")

            for _, row in df_f.sort_values("base").iterrows():
                row_vals = [_format_float(row[c]) for c in cols]
                lines.append("    " + " & ".join([str(row["base"])] + row_vals) + " \\\\")

            lines.append("    \\hline")
            lines.append("  \\end{tabular}")
            lines.append(f"  \\caption{{{caption}}}")
            lines.append(f"  \\label{{{label}}}")
            lines.append("\\end{table}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate LaTeX tables from sensitivity summary.")
    parser.add_argument("--summary", required=True, help="Path to summary.csv from sensitivity runner")
    parser.add_argument("--out", default=None, help="Output .tex file (default: summary directory / tables.tex)")
    args = parser.parse_args()

    summary_path = args.summary
    if not os.path.isabs(summary_path) and summary_path.startswith("results"):
        summary_path = os.path.join(_ROOT, summary_path)

    out_path = args.out
    if out_path is None:
        out_dir = os.path.dirname(summary_path)
        out_path = os.path.join(out_dir, "tables.tex")
    elif not os.path.isabs(out_path) and out_path.startswith("results"):
        out_path = os.path.join(_ROOT, out_path)

    make_tables(summary_path, out_path)


if __name__ == "__main__":
    main()
