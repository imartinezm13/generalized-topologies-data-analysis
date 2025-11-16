from __future__ import annotations

import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs, make_moons, make_circles
from typing import Tuple, Dict, Any


def load_pisa_xlsm(path: str) -> pd.DataFrame:
    """
    Load the PISA Excel workbook. Mirror of main.load_data for future reuse.
    """
    return pd.read_excel(path)


def load_gapminder_csv(path: str) -> pd.DataFrame:
    """
    Load a Gapminder-like CSV with columns such as country, continent, gdpPercap, lifeExp, pop.
    The function returns the raw DataFrame; downstream code can derive tags.
    """
    return pd.read_csv(path)


def load_gapminder_pkg() -> pd.DataFrame:
    """
    Load the gapminder dataset. Prefers the 'gapminder' package; falls back to plotly.express if unavailable.
    Returns a DataFrame with at least: ['country','continent','year','lifeExp','pop','gdpPercap'].
    """
    # Try the gapminder package first
    try:
        from gapminder import gapminder as gm_df  # type: ignore
        return gm_df.copy()
    except Exception:
        try:
            # Fallback: use Plotly's built-in gapminder dataset
            import plotly.express as px  # type: ignore
            gm_df = px.data.gapminder()
            return gm_df.copy()
        except Exception as e:
            raise ImportError(
                "Could not load gapminder data via 'gapminder' package or plotly.express."
            ) from e


def load_uci(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a classic UCI-like dataset via scikit-learn loaders.
    Returns (X_df, y_series).
    Supported: 'iris', 'wine', 'breast_cancer'.
    """
    name = name.lower()
    if name == 'iris':
        b = datasets.load_iris()
    elif name == 'wine':
        b = datasets.load_wine()
    elif name in ('breast_cancer', 'cancer'):
        b = datasets.load_breast_cancer()
    else:
        raise ValueError("Unsupported UCI dataset: {name}")
    X = pd.DataFrame(b.data, columns=b.feature_names)
    y = pd.Series(b.target, name='label')
    return X, y


def make_synthetic(kind: str = 'blobs', n_samples: int = 80, noise: float = 0.1, random_state: int | None = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a synthetic dataset. Returns (X_df, cluster_labels).
    kind in {'blobs','moons','circles'}.
    """
    kind = kind.lower()
    if kind == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0 + noise, random_state=random_state)
        cols = ['x1', 'x2'] if X.shape[1] == 2 else [f'x{i+1}']*X.shape[1]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name='cluster')
    elif kind == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return pd.DataFrame(X, columns=['x1', 'x2']), pd.Series(y, name='cluster')
    elif kind == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        return pd.DataFrame(X, columns=['x1', 'x2']), pd.Series(y, name='cluster')
    else:
        raise ValueError("Unsupported synthetic kind: {kind}")
