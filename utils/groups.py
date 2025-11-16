import pandas as pd
import numpy as np
from typing import Dict, List, Iterable, Optional, Tuple

def build_groups_from_categorical(df: pd.DataFrame, column: str) -> Dict[str, List[str]]:
    """
    Build group definitions from a categorical column.

    Returns a dict mapping category -> list of row indices (as strings).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    groups: Dict[str, List[str]] = {}
    for val, idxs in df.groupby(column).groups.items():
        groups[str(val)] = [str(i) for i in idxs]
    return groups


def build_groups_from_quantiles(
    df: pd.DataFrame,
    column: str,
    q: int = 4,
    labels: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Build group definitions by binning a numeric column into quantiles.
    Default is quartiles, labeled Low/MedLow/MedHigh/High.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if labels is None:
        if q == 3:
            labels = ["Low", "Medium", "High"]
        elif q == 4:
            labels = ["Low", "MedLow", "MedHigh", "High"]
        else:
            labels = [f"Q{i+1}" for i in range(q)]
    bins = pd.qcut(df[column], q=q, labels=labels, duplicates='drop')
    groups: Dict[str, List[str]] = {}
    for val, idxs in bins.groupby(bins).groups.items():
        groups[str(val)] = [str(i) for i in idxs]
    return groups


def composite_index(df: pd.DataFrame, columns: Iterable[str], name: str = "composite_index") -> pd.Series:
    """
    Create a simple composite index by z-scoring the provided columns and averaging.
    Returns a pandas Series with the composite scores.
    """
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError("No valid columns provided for composite index")
    z = (df[cols] - df[cols].mean()) / (df[cols].std(ddof=0).replace(0, 1))
    return z.mean(axis=1).rename(name)


def build_group_definitions(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    quantile_cols: Optional[List[str]] = None,
    q: int = 4,
    composite_specs: Optional[List[Tuple[List[str], str]]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a dictionary of group-definition families from available columns.

    Returns a dict mapping family_name -> { group_name -> [row indices as strings] }.
    """
    families: Dict[str, Dict[str, List[str]]] = {}

    # Categorical families
    for col in (categorical_cols or []):
        if col in df.columns and (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])):
            families[col] = build_groups_from_categorical(df, col)

    # Quantile-binned numeric families
    for col in (quantile_cols or []):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            families[f"{col}_quantiles"] = build_groups_from_quantiles(df, col, q=q)

    # Composite families
    for cols, fam_name in (composite_specs or []):
        try:
            comp = composite_index(df, cols, name=fam_name)
            tmp = df.copy()
            tmp[fam_name] = comp
            families[fam_name] = build_groups_from_quantiles(tmp, fam_name, q=q)
        except ValueError:
            # Skip invalid composite spec silently
            continue

    return families

