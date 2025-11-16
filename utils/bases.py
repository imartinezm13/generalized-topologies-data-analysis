import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils.data_utils import stand, positions_less_or_equal, avg_data
from utils.regression import multi_regression, evaluate_polynomial
from utils.regression_robust import optimal_point_robust as optimal_point

from Paquete.convertir_a_arbol import convertir_a_Tree
from Paquete.obtener_subarboles import asignar_nombres, obtener_subarboles
from Paquete.obtener_n_subarboles import obtener_n_subarboles
from Paquete.calcular_sn import calcular_sn
from Paquete.base_topologica import base_topologica
from Paquete.obtener_maximales import obtener_maximales


def build_base(df_scaled, linkage_method: str = 'single', distance_metric: str = 'euclidean'):
    """Build the base topological structure from scaled data.

    Parameters:
    - df_scaled: array-like, scaled numeric data
    - linkage_method: str, hierarchical linkage criterion (e.g., 'single', 'complete', 'average', 'ward')
    - distance_metric: str, distance metric for pdist (e.g., 'euclidean', 'cityblock', 'cosine', 'correlation')
    """
    if linkage_method == 'ward':
        dendrogram = linkage(df_scaled, method='ward')
    else:
        condensed = pdist(df_scaled, metric=distance_metric)
        dendrogram = linkage(condensed, method=linkage_method)

    result_tree = convertir_a_Tree(dendrogram, leaf_names=range(len(df_scaled)))
    asignar_nombres(result_tree)
    all_subtrees = obtener_subarboles(result_tree)
    n_subtrees = obtener_n_subarboles(all_subtrees, len(df_scaled))
    maximals = obtener_maximales(n_subtrees)
    sn = calcular_sn(maximals)
    base = base_topologica(sn, maximals)
    return base


def build_gen_base_1(Base, df, distance_metric: str = 'euclidean', target_r2: float = 0.99):
    """Build the gen_base_1 from the base structure."""
    Base_int = [[int(item) for item in subset] for subset in Base]

    min_values = []
    for subset in Base_int:
        subset_df = df.iloc[subset]
        min_row = subset_df.min()
        min_values.append(min_row)

    min_values = pd.DataFrame(min_values)
    df_min = stand(min_values).astype(float)
    distances = pdist(df_min, metric=distance_metric)
    distance_matrix = squareform(distances)
    distance_df = pd.DataFrame(distance_matrix, index=df_min.index, columns=df_min.index)
    triangular_df = pd.DataFrame(np.triu(distance_df.values), index=df_min.index, columns=df_min.index)

    dist_values = triangular_df.values
    vector = dist_values.flatten()
    vec = np.sort(vector[vector > 0])

    if len(vec) == 0:
        # No pairwise distances > 0; return Base unchanged
        return Base
    y = np.array([i for i in range(1, len(vec) + 1)])
    r = target_r2
    poly, _ = multi_regression(vec, y, r)
    squared_differences = (y - [evaluate_polynomial(vec[i], poly)[0] for i in range(len(vec))]) ** 2
    try:
        x_min = optimal_point(vec, squared_differences)
    except Exception:
        # Fallback: discrete argmin
        x_min = float(vec[np.argmin(squared_differences)])

    positions = positions_less_or_equal(dist_values, x_min)
    M = set([int(index) for pos in positions for index in pos])
    M = list(M)
    new_base = [Base[pos[0]] + Base[pos[1]] for pos in positions]
    gen_base_1 = [Base[i] for i in range(len(Base)) if i not in M] + new_base
    return gen_base_1


def build_gen_base_2(Base, df_scaled, df, distance_metric: str = 'euclidean', target_r2: float = 0.99, show_progress: bool = False):
    """Constructs the gen_base_2 set by merging base elements based on distance and regression analysis.

    If show_progress is True, a tqdm progress bar is displayed over thresholds.
    """
    index_base = [np.array(group, dtype=int) for group in Base]
    A = avg_data(index_base, df_scaled).astype(float)
    distances = pdist(A, metric=distance_metric)
    distance_matrix = squareform(distances)
    triangular = np.triu(distance_matrix)
    vector = triangular.flatten()
    Vec = np.sort(vector[vector > 0])

    W_i = []
    iterable = tqdm(Vec, desc="gen_base_2 thresholds", leave=False) if show_progress else Vec
    for threshold in iterable:
        positions = positions_less_or_equal(triangular, threshold)
        if positions.size == 0:
            W_i.append(0)
            continue
        merged_indices = list(set(positions.flatten()))
        new_base = [np.concatenate((index_base[pos[0]], index_base[pos[1]])) for pos in positions]
        current_base = new_base
        if not current_base:
            W_i.append(0)
            continue
        A_for_new_base = avg_data(current_base, df).astype(float)
        w = 0
        for l in range(len(A_for_new_base)):
            indices = np.array(current_base[l], dtype=int)
            diff_vectors = A_for_new_base[l] - df.iloc[indices].values
            w += np.sum(np.concatenate(np.square(diff_vectors)))
        W_i.append(w)

    if len(Vec) == 0:
        return Base
    r = target_r2
    poly1, _ = multi_regression(Vec, W_i, r)
    squared_differences = (np.array(W_i) - np.array([evaluate_polynomial(v, poly1)[0] for v in Vec])) ** 2
    try:
        x_min = optimal_point(Vec, squared_differences)
    except Exception:
        x_min = float(Vec[np.argmin(squared_differences)])

    positions = positions_less_or_equal(triangular, x_min)
    merged_indices = []
    for i in range(len(positions)):
        merged_indices += list(positions[i])
    merged_indices = list(set(merged_indices))

    new_base = []
    for i in range(len(positions)):
        new_base.append(Base[positions[i][0]] + Base[positions[i][1]])

    gen_base_2 = Base + new_base
    for i in range(len(merged_indices)):
        gen_base_2.remove(Base[merged_indices[i]])

    return gen_base_2


def kmeans_clusters_from_columns(df: pd.DataFrame, feature_columns, n_clusters: int = 4, random_state: int | None = 42):
    """Generic KMeans clustering groups indices by label, returning groups of string indices."""
    scaler = StandardScaler()
    X = df[feature_columns]
    X = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return [[str(i) for i in np.where(labels == cluster_id)[0]] for cluster_id in range(n_clusters)]
