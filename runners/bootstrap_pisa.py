import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

# Ensure project root is on sys.path when running from runners/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force non-interactive matplotlib backend for any downstream imports
os.environ.setdefault("MPLBACKEND", "Agg")

from main import load_data, preprocess_data
from utils.data_utils import avg_data, positions_less_or_equal
from utils.regression import evaluate_polynomial, multi_regression
from utils.regression_robust import optimal_point_robust as optimal_point

from Paquete.convertir_a_arbol import convertir_a_Tree
from Paquete.obtener_subarboles import asignar_nombres, obtener_subarboles
from Paquete.obtener_n_subarboles import obtener_n_subarboles
from Paquete.calcular_sn import calcular_sn
from Paquete.base_topologica import base_topologica
from Paquete.obtener_maximales import obtener_maximales


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Wilson score interval for a proportion; stable near 0/1."""
    if total == 0:
        return 0.0, 0.0, 1.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / total
    denom = 1 + z * z / total
    center = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return phat, lower, upper


def percentile_interval(values: List[float], confidence: float = 0.95) -> Optional[Tuple[float, float]]:
    """Percentile interval for continuous statistics like v_k."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    alpha = 1 - confidence
    lower = float(np.percentile(clean, 100 * alpha / 2))
    upper = float(np.percentile(clean, 100 * (1 - alpha / 2)))
    return lower, upper


def select_pair(positions: np.ndarray, distances: np.ndarray) -> Optional[Tuple[int, int]]:
    """Pick the closest pair (by distance) among those under threshold."""
    if positions.size == 0:
        return None
    best_pair = None
    best_dist = None
    for i, j in positions:
        dist = distances[int(i), int(j)]
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pair = (int(i), int(j))
    return best_pair


def sn_cluster_to_original(
    base: List[Sequence[int]],
    max_sn_leaf: Optional[int],
    sample_indices: Sequence[int],
) -> Optional[List[int]]:
    """Given a topological base and the leaf with max S_n, return the original IDs in its block."""
    if max_sn_leaf is None or base is None:
        return None

    leaf = int(max_sn_leaf)
    n = len(sample_indices)

    # 1) Find the block in the base that contains the leaf
    cluster_local = None
    for subset in base:
        subset_int = [int(x) for x in subset]
        if leaf in subset_int:
            cluster_local = subset_int
            break

    if cluster_local is None:
        return None

    # 2) Map to original IDs (taking care with the same index trick used in map_*_to_original)
    original_ids = set()
    for idx in cluster_local:
        val = int(idx)
        if val >= n and val - 1 < n:
            val -= 1
        if 0 <= val < n:
            original_ids.add(int(sample_indices[val]))

    if not original_ids:
        return None

    # Return sorted list so it can be serialized in JSON/CSV
    return sorted(original_ids)

def sn_node_leaves_from_linkage(Z, node_id: Optional[int]) -> Optional[List[int]]:
    """
    Given a SciPy linkage matrix Z and a node_id (leaf < n, internal >= n),
    return the list of leaf indices (0..n-1) under that node.
    """
    if node_id is None:
        return None

    node_id = int(node_id)
    root, nodes = to_tree(Z, rd=True)

    # nodes list is indexed by node.id. Typical IDs: 0..n-1 leaves, n..2n-2 internals.
    if node_id < 0 or node_id >= len(nodes):
        # If you suspect 1-based indexing, try a fallback:
        if 0 <= node_id - 1 < len(nodes):
            node_id = node_id - 1
        else:
            return None

    node = nodes[node_id]
    return node.pre_order()  # returns leaf ids


def map_leaf_to_original(leaf_idx: Optional[int], sample_indices: Sequence[int]) -> Optional[int]:
    """Convert a leaf index (bootstrap position) to original row id."""
    if leaf_idx is None:
        return None
    n = len(sample_indices)
    idx = int(leaf_idx)
    # Heuristic to handle possible 1-based labels
    if idx >= n and idx - 1 < n:
        idx = idx - 1
    if idx < 0 or idx >= n:
        return None
    return int(sample_indices[idx])


def map_pair_to_original(pair: Optional[Tuple[int, int]], base_int: List[List[int]], sample_indices: Sequence[int]) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Map a pair of base indices to the underlying original row ids."""
    if pair is None:
        return None
    i, j = pair
    n = len(sample_indices)
    def _normalize(indices: List[int]) -> Tuple[int, ...]:
        normalized = []
        for idx in indices:
            val = int(idx)
            if val >= n and val - 1 < n:
                val = val - 1
            normalized.append(int(sample_indices[val]))
        return tuple(sorted(normalized))

    subset_i = _normalize(base_int[i])
    subset_j = _normalize(base_int[j])
    ordered = tuple(sorted([subset_i, subset_j]))
    return ordered


def merged_set_to_original(
    base_int: List[List[int]],
    positions: np.ndarray,
    sample_indices: Sequence[int],
) -> Optional[List[int]]:
    """Return the union of all elements merged at the given positions in original ids."""
    if positions is None or positions.size == 0:
        return None

    n = len(sample_indices)
    merged_original_ids = set()

    for b_idx1, b_idx2 in positions:
        b_idx1 = int(b_idx1)
        b_idx2 = int(b_idx2)
        if b_idx1 < 0 or b_idx1 >= len(base_int) or b_idx2 < 0 or b_idx2 >= len(base_int):
            continue

        local_indices = list(base_int[b_idx1]) + list(base_int[b_idx2])

        for local_idx in local_indices:
            val = int(local_idx)
            if val >= n and val - 1 < n:
                val -= 1
            if 0 <= val < n:
                merged_original_ids.add(int(sample_indices[val]))

    if not merged_original_ids:
        return None

    return sorted(merged_original_ids)


def jaccard_similarity(a: Optional[Sequence[int]], b: Optional[Sequence[int]]) -> float:
    """Jaccard similarity between two index sets; 0 if either is None."""
    if a is None or b is None:
        return 0.0
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    union = len(A | B)
    if union == 0:
        return 0.0
    return len(A & B) / union


def compute_base_with_metadata(
    df_scaled: np.ndarray, linkage_method: str, distance_metric: str
) -> Tuple[List[List[int]], Optional[int], np.ndarray]:
    """Build base and return the leaf achieving max S_n, plus the linkage matrix Z."""
    if linkage_method == "ward":
        Z = linkage(df_scaled, method="ward")
    else:
        condensed = pdist(df_scaled, metric=distance_metric)
        Z = linkage(condensed, method=linkage_method)

    result_tree = convertir_a_Tree(Z, leaf_names=range(len(df_scaled)))
    asignar_nombres(result_tree)
    all_subtrees = obtener_subarboles(result_tree)
    n_subtrees = obtener_n_subarboles(all_subtrees, len(df_scaled))
    maximals = obtener_maximales(n_subtrees)
    sn = calcular_sn(maximals)
    base = base_topologica(sn, maximals)

    max_sn_node = None
    if sn:
        try:
            max_sn_node = int(max(sn, key=lambda x: x[1])[0])  # node/subtree id
        except Exception:
            max_sn_node = None

    return base, max_sn_node, Z


def silhouette_for_clusters(df: pd.DataFrame, clusters: List[Sequence[int]]) -> Optional[float]:
    """
    Compute silhouette score for a clustering described by list-of-indices clusters.
    Returns None if fewer than 2 non-empty clusters.
    """
    # Deduplicate and drop empty clusters
    cluster_lists = [sorted(set(int(i) for i in c)) for c in clusters]
    cluster_lists = [c for c in cluster_lists if len(c) > 0]
    if len(cluster_lists) < 2:
        return None

    X = df.to_numpy()
    n = len(X)
    if n == 0:
        return None

    # Map point -> cluster id for quick lookup
    point_to_cluster = {}
    for cid, clist in enumerate(cluster_lists):
        for idx in clist:
            point_to_cluster[idx] = cid

    # If some points are missing from all clusters, silhouette is undefined
    if len(point_to_cluster) < n:
        return None

    dist_matrix = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = np.zeros(n)

    for i in range(n):
        cid = point_to_cluster.get(i)
        if cid is None:
            return None
        same_cluster = cluster_lists[cid]
        if len(same_cluster) <= 1:
            a_i = 0.0
        else:
            a_i = np.mean([dist_matrix[i, j] for j in same_cluster if j != i])

        b_i = None
        for other_cid, other_cluster in enumerate(cluster_lists):
            if other_cid == cid or len(other_cluster) == 0:
                continue
            mean_dist = np.mean([dist_matrix[i, j] for j in other_cluster])
            if b_i is None or mean_dist < b_i:
                b_i = mean_dist

        if b_i is None:
            return None

        denom = max(a_i, b_i)
        s[i] = 0.0 if denom == 0 else (b_i - a_i) / denom

    return float(np.mean(s))


def build_gen_base_1_with_meta(Base, df, distance_metric: str, target_r2: float):
    """Run build_gen_base_1 and also keep v_k (threshold) and chosen pair."""
    Base_int = [[int(item) for item in subset] for subset in Base]
    min_values = []
    for subset in Base_int:
        subset_df = df.iloc[subset]
        min_row = subset_df.min()
        min_values.append(min_row)

    min_values = pd.DataFrame(min_values)
    df_min = preprocess_data(min_values).astype(float)
    distances = pdist(df_min, metric=distance_metric)
    distance_matrix = squareform(distances)
    triangular_df = np.triu(distance_matrix)

    dist_values = triangular_df
    vector = dist_values.flatten()
    vec = np.sort(vector[vector > 0])

    if len(vec) == 0:
        return Base, None, None, Base_int

    y = np.arange(1, len(vec) + 1)
    r = target_r2
    poly, _ = multi_regression(vec, y, r)
    squared_differences = (y - [evaluate_polynomial(vec[i], poly)[0] for i in range(len(vec))]) ** 2
    try:
        x_min = optimal_point(vec, squared_differences)
    except Exception:
        x_min = float(vec[np.argmin(squared_differences)])

    positions = positions_less_or_equal(dist_values, x_min)
    chosen_pair = select_pair(positions, dist_values)

    M = set([int(index) for pos in positions for index in pos])
    new_base = [Base[pos[0]] + Base[pos[1]] for pos in positions]
    gen_base_1 = [Base[i] for i in range(len(Base)) if i not in M] + new_base

    return gen_base_1, x_min, chosen_pair, Base_int, positions


def build_gen_base_2_with_meta(Base, df_scaled, df, distance_metric: str, target_r2: float):
    """Run build_gen_base_2 and keep v_k and centroid pair."""
    index_base = [np.array(group, dtype=int) for group in Base]
    A = avg_data(index_base, df_scaled).astype(float)
    distances = pdist(A, metric=distance_metric)
    distance_matrix = squareform(distances)
    triangular = np.triu(distance_matrix)
    vector = triangular.flatten()
    Vec = np.sort(vector[vector > 0])

    W_i = []
    for threshold in Vec:
        positions = positions_less_or_equal(triangular, threshold)
        if positions.size == 0:
            W_i.append(0)
            continue
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
        return Base, None, None, [list(g) for g in index_base]

    r = target_r2
    poly1, _ = multi_regression(Vec, W_i, r)
    squared_differences = (np.array(W_i) - np.array([evaluate_polynomial(v, poly1)[0] for v in Vec])) ** 2
    try:
        x_min = optimal_point(Vec, squared_differences)
    except Exception:
        x_min = float(Vec[np.argmin(squared_differences)])

    positions = positions_less_or_equal(triangular, x_min)
    chosen_pair = select_pair(positions, triangular)

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

    base_int = [list(map(int, g)) for g in Base]
    return gen_base_2, x_min, chosen_pair, base_int, positions


def run_single(df_numeric: pd.DataFrame,
               df_scaled: np.ndarray,
               sample_indices: np.ndarray,
               linkage_method: str,
               distance_metric: str,
               target_r2: float):
    """Execute one run (original or bootstrap) and capture selections."""
    # 1) Base maximal + S_n winner
    base, max_sn_node, Z = compute_base_with_metadata(df_scaled, linkage_method, distance_metric)
    base_max_original = map_leaf_to_original(max_sn_node, sample_indices)

    sn_cluster_original = sn_cluster_to_original(base, max_sn_node, sample_indices)
    sn_cluster_size = len(sn_cluster_original) if sn_cluster_original is not None else None

    # 2) Gen Base 1 (column-min representatives)
    gen_base_1, vk1, min_pair, base_int_1, positions_1 = build_gen_base_1_with_meta(
        base, df_numeric, distance_metric, target_r2
    )
    min_pair_original = map_pair_to_original(min_pair, base_int_1, sample_indices)
    gen_base_1_merged_set = merged_set_to_original(base_int_1, positions_1, sample_indices)

    # 3) Gen Base 2 (centroid representatives)
    gen_base_2, vk2, centroid_pair, base_int_2, positions_2 = build_gen_base_2_with_meta(
        base, df_scaled, df_numeric, distance_metric, target_r2
    )
    centroid_pair_original = map_pair_to_original(centroid_pair, base_int_2, sample_indices)
    gen_base_2_merged_set = merged_set_to_original(base_int_2, positions_2, sample_indices)

    # 4) Silhouette on the base
    silhouette_score = silhouette_for_clusters(df_numeric, base) if base else None

    return {
        "base_max_node": max_sn_node,
        "base_max_original": base_max_original,
        "sn_cluster_original": sn_cluster_original,
        "sn_cluster_size": sn_cluster_size,
        "gen_base_1_vk": vk1,
        "gen_base_1_pair": min_pair,
        "gen_base_1_pair_original": min_pair_original,
        "gen_base_1_merged_set": gen_base_1_merged_set,
        "gen_base_2_vk": vk2,
        "gen_base_2_pair": centroid_pair,
        "gen_base_2_pair_original": centroid_pair_original,
        "gen_base_2_merged_set": gen_base_2_merged_set,
        "base": base,
        "gen_base_1": gen_base_1,
        "gen_base_2": gen_base_2,
        "silhouette": silhouette_score,
    }


def bootstrap_pisa(bootstraps: int,
                   confidence: float,
                   linkage_method: str,
                   distance_metric: str,
                   target_r2: float,
                   random_seed: Optional[int],
                   output_root: Path):
    """Run bootstrap over the PISA pipeline and compute probabilities/CIs."""
    rng = np.random.default_rng(random_seed)

    df_original = load_data()
    qualitative = ['Country', 'Type of Economy', 'Population Density', 'Type of Government', 'Continent']
    df_numeric = df_original.drop(columns=qualitative)
    N_total = len(df_numeric)

    output_root.mkdir(parents=True, exist_ok=True)
    replicates_path = output_root / "replicates.csv"
    baseline_path = output_root / "baseline.json"

    # If a baseline already exists (resume), load it; otherwise compute and persist it.
    if baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text())
        baseline = baseline_payload["baseline"]
        baseline_config = baseline_payload.get("config", {})
    else:
        df_scaled = preprocess_data(df_numeric)
        baseline_indices = np.arange(len(df_numeric))
        baseline_run = run_single(df_numeric, df_scaled, baseline_indices, linkage_method, distance_metric, target_r2)
        baseline = {
            "base_max_original": baseline_run["base_max_original"],
            "sn_cluster_original": baseline_run["sn_cluster_original"],
            "sn_cluster_size": baseline_run["sn_cluster_size"],
            "gen_base_1_vk": baseline_run["gen_base_1_vk"],
            "gen_base_1_pair_original": baseline_run["gen_base_1_pair_original"],
            "gen_base_1_merged_set": baseline_run["gen_base_1_merged_set"],
            "gen_base_2_vk": baseline_run["gen_base_2_vk"],
            "gen_base_2_pair_original": baseline_run["gen_base_2_pair_original"],
            "gen_base_2_merged_set": baseline_run["gen_base_2_merged_set"],
            "silhouette": baseline_run["silhouette"],
        }
        baseline_config = {
            "linkage_method": linkage_method,
            "distance_metric": distance_metric,
            "target_r2": target_r2,
            "random_seed": random_seed,
        }
        baseline_payload = {"baseline": baseline, "config": baseline_config}
        baseline_path.write_text(json.dumps(baseline_payload, indent=2))

    base_matches = 0
    gen1_pair_matches = 0
    gen2_pair_matches = 0
    vk1_values = []
    vk2_values = []
    silhouette_values = []
    records = []
    gen1_jaccards = []
    gen2_jaccards = []
    # Treat a bootstrap as a "match" if the merged set overlaps the baseline above this Jaccard level
    jaccard_threshold = 0.5

    # If partial results exist, warm-start counts from disk.
    start_iter = 0
    if replicates_path.exists():
        existing = pd.read_csv(replicates_path)
        start_iter = len(existing)
        if start_iter > 0:
            print(f"[bootstrap] found {start_iter} existing replicates; continuing from there.")
            base_matches = int((existing["base_max_original"] == baseline["base_max_original"]).sum())
            j1_existing = existing["gen_base_1_jaccard_vs_baseline"].dropna().tolist()
            j2_existing = existing["gen_base_2_jaccard_vs_baseline"].dropna().tolist()
            gen1_jaccards.extend(j1_existing)
            gen2_jaccards.extend(j2_existing)
            gen1_pair_matches = int((existing["gen_base_1_jaccard_vs_baseline"] > jaccard_threshold).sum())
            gen2_pair_matches = int((existing["gen_base_2_jaccard_vs_baseline"] > jaccard_threshold).sum())
            vk1_values.extend(existing["gen_base_1_vk"].dropna().tolist())
            vk2_values.extend(existing["gen_base_2_vk"].dropna().tolist())
            silhouette_values.extend(existing["silhouette"].dropna().tolist())

    # If the user asks for fewer iterations than already exist, reuse them.
    total_reps = max(bootstraps, start_iter)

    def append_record(record: Dict):
        """Persist a single replicate row immediately."""
        file_exists = replicates_path.exists()
        pd.DataFrame([record]).to_csv(
            replicates_path, mode="a", header=not file_exists, index=False
        )

    for b in range(start_iter, total_reps):
        if b == start_iter or (b + 1) % max(1, total_reps // 10) == 0 or b == total_reps - 1:
            print(f"[bootstrap] iteration {b+1}/{total_reps}")
        # Keep the original row ids so we can map selections back after resampling.
        sample_idx = rng.integers(0, len(df_numeric), size=len(df_numeric))
        distinct_countries = int(len(np.unique(sample_idx)))
        missing_countries = int(N_total - distinct_countries)

        df_boot = df_numeric.iloc[sample_idx].reset_index(drop=True)
        df_scaled_boot = preprocess_data(df_boot)

        run = run_single(df_boot, df_scaled_boot, sample_idx, linkage_method, distance_metric, target_r2)

        base_matches += int(run["base_max_original"] == baseline["base_max_original"])

        # Compare merged sets against baseline using Jaccard to allow partial overlap
        j1 = jaccard_similarity(
            baseline["gen_base_1_merged_set"],
            run["gen_base_1_merged_set"],
        )
        j2 = jaccard_similarity(
            baseline["gen_base_2_merged_set"],
            run["gen_base_2_merged_set"],
        )

        gen1_jaccards.append(j1)
        gen2_jaccards.append(j2)

        gen1_pair_matches += int(j1 > jaccard_threshold)
        gen2_pair_matches += int(j2 > jaccard_threshold)

        vk1_values.append(run["gen_base_1_vk"])
        vk2_values.append(run["gen_base_2_vk"])
        silhouette_values.append(run["silhouette"])

        record = {
            "bootstrap": b,
            "base_max_original": run["base_max_original"],
            "sn_cluster_size": run["sn_cluster_size"],
            "distinct_countries": distinct_countries,
            "missing_countries": missing_countries,
            "gen_base_1_vk": run["gen_base_1_vk"],
            "gen_base_1_pair_original": run["gen_base_1_pair_original"],
            "gen_base_1_merged_set": run["gen_base_1_merged_set"],
            "gen_base_2_vk": run["gen_base_2_vk"],
            "gen_base_2_pair_original": run["gen_base_2_pair_original"],
            "gen_base_2_merged_set": run["gen_base_2_merged_set"],
            "silhouette": run["silhouette"],
            "gen_base_1_jaccard_vs_baseline": j1,
            "gen_base_2_jaccard_vs_baseline": j2,
        }
        records.append(record)
        append_record(record)

    # All statistics use the total number of available replicates (existing + new)
    base_p, base_lo, base_hi = wilson_interval(base_matches, total_reps, confidence)
    gen1_p, gen1_lo, gen1_hi = wilson_interval(gen1_pair_matches, total_reps, confidence)
    gen2_p, gen2_lo, gen2_hi = wilson_interval(gen2_pair_matches, total_reps, confidence)

    vk1_ci = percentile_interval(vk1_values, confidence)
    vk2_ci = percentile_interval(vk2_values, confidence)
    silhouette_ci = percentile_interval([v for v in silhouette_values if v is not None], confidence)

    summary = {
        "config": {
            "bootstraps": total_reps,
            "confidence": confidence,
            "linkage_method": baseline_config.get("linkage_method", linkage_method),
            "distance_metric": baseline_config.get("distance_metric", distance_metric),
            "target_r2": baseline_config.get("target_r2", target_r2),
            "random_seed": baseline_config.get("random_seed", random_seed),
        },
        "baseline": {
            "base_max_original": baseline["base_max_original"],
            "gen_base_1_vk": baseline["gen_base_1_vk"],
            "gen_base_1_pair_original": baseline["gen_base_1_pair_original"],
            "gen_base_2_vk": baseline["gen_base_2_vk"],
            "gen_base_2_pair_original": baseline["gen_base_2_pair_original"],
            "silhouette": baseline["silhouette"],
        },
        "probabilities": {
            "base_max_match": {"p": base_p, "ci_lower": base_lo, "ci_upper": base_hi},
            "gen_base_1_pair_match": {"p": gen1_p, "ci_lower": gen1_lo, "ci_upper": gen1_hi},
            "gen_base_2_pair_match": {"p": gen2_p, "ci_lower": gen2_lo, "ci_upper": gen2_hi},
        },
        "similarities": {
            "jaccard_threshold": jaccard_threshold,
            "gen_base_1_mean_jaccard": float(np.mean(gen1_jaccards)) if gen1_jaccards else None,
            "gen_base_2_mean_jaccard": float(np.mean(gen2_jaccards)) if gen2_jaccards else None,
        },
        "vk_intervals": {
            "gen_base_1_vk": vk1_ci,
            "gen_base_2_vk": vk2_ci,
        },
        "silhouette_interval": silhouette_ci,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame.from_records(records).to_csv(output_root / "replicates.csv", index=False)

    return summary


def regenerate_summary(output_root: Path, confidence: float = 0.95, jaccard_threshold: float = 0.5):
    """
    Rebuild summary.json from an existing baseline.json + replicates.csv.
    Useful after an interrupted run that already wrote partial results.
    """
    baseline_path = output_root / "baseline.json"
    replicates_path = output_root / "replicates.csv"

    if not baseline_path.exists() or not replicates_path.exists():
        raise FileNotFoundError("baseline.json or replicates.csv not found; run bootstrap first.")

    baseline_payload = json.loads(baseline_path.read_text())
    baseline = baseline_payload["baseline"]
    baseline_config = baseline_payload.get("config", {})
    df = pd.read_csv(replicates_path)
    total_reps = len(df)
    if total_reps == 0:
        raise ValueError("replicates.csv is empty; nothing to summarize.")

    base_matches = int((df["base_max_original"] == baseline["base_max_original"]).sum())
    gen1_jaccards = df["gen_base_1_jaccard_vs_baseline"].dropna().tolist()
    gen2_jaccards = df["gen_base_2_jaccard_vs_baseline"].dropna().tolist()
    gen1_pair_matches = int((df["gen_base_1_jaccard_vs_baseline"] > jaccard_threshold).sum())
    gen2_pair_matches = int((df["gen_base_2_jaccard_vs_baseline"] > jaccard_threshold).sum())
    vk1_values = df["gen_base_1_vk"].dropna().tolist()
    vk2_values = df["gen_base_2_vk"].dropna().tolist()
    silhouette_values = df["silhouette"].dropna().tolist()

    base_p, base_lo, base_hi = wilson_interval(base_matches, total_reps, confidence)
    gen1_p, gen1_lo, gen1_hi = wilson_interval(gen1_pair_matches, total_reps, confidence)
    gen2_p, gen2_lo, gen2_hi = wilson_interval(gen2_pair_matches, total_reps, confidence)

    vk1_ci = percentile_interval(vk1_values, confidence)
    vk2_ci = percentile_interval(vk2_values, confidence)
    silhouette_ci = percentile_interval([v for v in silhouette_values if v is not None], confidence)

    summary = {
        "config": {
            "bootstraps": total_reps,
            "confidence": confidence,
            "linkage_method": baseline_config.get("linkage_method"),
            "distance_metric": baseline_config.get("distance_metric"),
            "target_r2": baseline_config.get("target_r2"),
            "random_seed": baseline_config.get("random_seed"),
        },
        "baseline": baseline,
        "probabilities": {
            "base_max_match": {"p": base_p, "ci_lower": base_lo, "ci_upper": base_hi},
            "gen_base_1_pair_match": {"p": gen1_p, "ci_lower": gen1_lo, "ci_upper": gen1_hi},
            "gen_base_2_pair_match": {"p": gen2_p, "ci_lower": gen2_lo, "ci_upper": gen2_hi},
        },
        "similarities": {
            "jaccard_threshold": jaccard_threshold,
            "gen_base_1_mean_jaccard": float(np.mean(gen1_jaccards)) if gen1_jaccards else None,
            "gen_base_2_mean_jaccard": float(np.mean(gen2_jaccards)) if gen2_jaccards else None,
        },
        "vk_intervals": {
            "gen_base_1_vk": vk1_ci,
            "gen_base_2_vk": vk2_ci,
        },
        "silhouette_interval": silhouette_ci,
    }

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Bootstrap robustness for PISA bases.")
    parser.add_argument("--bootstraps", type=int, default=100, help="Number of bootstrap replicates.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals.")
    parser.add_argument("--linkage", default="single", choices=["single", "complete", "average", "ward"], help="Linkage method for dendrogram.")
    parser.add_argument("--metric", default="euclidean", choices=["euclidean", "cityblock", "cosine", "correlation"], help="Distance metric (ignored for ward).")
    parser.add_argument("--target-r2", type=float, default=0.99, help="Target R^2 for regression thresholding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap replicates.")
    parser.add_argument("--out", default="results/bootstrap", help="Output directory for summary and per-replicate CSV.")
    parser.add_argument("--regen-summary", action="store_true", help="Regenerate summary.json from existing results and exit.")

    args = parser.parse_args()
    output_root = Path(args.out)

    if args.regen_summary:
        summary = regenerate_summary(output_root, confidence=args.confidence)
    else:
        summary = bootstrap_pisa(
            bootstraps=args.bootstraps,
            confidence=args.confidence,
            linkage_method=args.linkage,
            distance_metric=args.metric,
            target_r2=args.target_r2,
            random_seed=args.seed,
            output_root=output_root,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
