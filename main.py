import pandas as pd
from utils.data_utils import stand, positions_less_or_equal, avg_data
from utils.regression import multi_regression, evaluate_polynomial, optimal_point
from utils.analysis import process_batch

from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage

from Paquete.convertir_a_arbol import convertir_a_Tree
from Paquete.obtener_subarboles import asignar_nombres, obtener_subarboles
from Paquete.obtener_n_subarboles import obtener_n_subarboles
from Paquete.calcular_sn import calcular_sn
from Paquete.base_topologica import base_topologica
from Paquete.obtener_maximales import obtener_maximales

DATA_PATH = "data/PISA_test_database.xlsm"

def load_data(path=DATA_PATH):
    """Load data from an Excel file."""
    df = pd.read_excel(path)
    return df

def preprocess_data(df):
    """Standardize the input DataFrame."""
    df_scaled = stand(df)
    return df_scaled

def build_base(df_scaled):
    """Build the base topological structure from scaled data."""

    dendrogram = linkage(df_scaled, method='single')  # Hierarchical clustering
    result_tree = convertir_a_Tree(dendrogram, leaf_names=range(len(df_scaled)))  # Dendrogram to tree
    asignar_nombres(result_tree)  # Assign names to nodes
    all_subtrees = obtener_subarboles(result_tree)  # Get all subtrees
    n_subtrees = obtener_n_subarboles(all_subtrees, len(df_scaled))  # Get n-subtrees
    maximals = obtener_maximales(n_subtrees)  # Find maximal subtrees
    sn = calcular_sn(maximals)  # Calculate Sn
    base = base_topologica(sn, maximals)  # Build base structure

    return base

def build_gen_base_1(Base, df):
    """Build the gen_base_1 from the base structure."""

    # Convert all elements in Base to integers for indexing
    Base_int = [[int(item) for item in subset] for subset in Base]

    min_values = []
    # For each subset in the base, find the row-wise minimum values in the DataFrame
    for subset in Base_int:
        subset_df = df.iloc[subset]  # Select rows corresponding to the subset
        min_row = subset_df.min()    # Find the minimum value for each column in the subset
        min_values.append(min_row)   # Store the minimum row

    # Create a DataFrame from the list of minimum rows
    min_values = pd.DataFrame(min_values)
    # Standardize the minimum values DataFrame
    df_min = stand(min_values).astype(float)
    # Compute pairwise Euclidean distances between the standardized minimum rows
    distances = pdist(df_min, metric='euclidean')
    # Convert the condensed distance matrix to a square form
    distance_matrix = squareform(distances)
    # Create a DataFrame for the distance matrix for easier indexing
    distance_df = pd.DataFrame(distance_matrix, index=df_min.index, columns=df_min.index)
    # Extract the upper triangular part of the distance matrix (to avoid duplicate pairs)
    triangular_df = pd.DataFrame(np.triu(distance_df.values), index=df_min.index, columns=df_min.index)

    # Get the values of the upper triangular distance matrix
    dist_values = triangular_df.values
    # Flatten the matrix to a vector
    vector = dist_values.flatten()
    # Sort the vector and remove zeros (self-distances)
    vec = np.sort(vector[vector > 0])

    # Create a sequence for regression (1, 2, ..., len(vec))
    y = np.array([i for i in range(1, len(vec)+1)])
    r = 0.99  # Regularization parameter for regression
    # Fit a polynomial regression to the sorted distances
    poly, _ = multi_regression(vec, y, r)
    # Compute squared differences between actual and predicted y values
    squared_differences = (y - [evaluate_polynomial(vec[i], poly)[0] for i in range(len(vec))])**2

    # Find the optimal threshold (x_min) that minimizes the squared differences
    x_min = optimal_point(vec, squared_differences)

    # Find positions in the distance matrix where the value is less than or equal to x_min
    positions = positions_less_or_equal(dist_values, x_min)
    # Collect all unique indices involved in these positions
    M = set([int(index) for pos in positions for index in pos])
    M = list(M)
    # For each position, merge the corresponding base subsets to form new base elements
    new_base = [Base[pos[0]] + Base[pos[1]] for pos in positions]
    # The new build_generalized_base_1 is the union of base elements not in M and the newly formed base elements
    build_generalized_base_1 = [Base[i] for i in range(len(Base)) if i not in M] + new_base

    return build_generalized_base_1

def build_gen_base_2(Base, df_scaled, df):
    """
    Constructs the gen_base_2 set by merging base elements based on distance and regression analysis.
    """
    # Convert each group in Base to a numpy array of integer indices
    index_base = [np.array(group, dtype=int) for group in Base]
    # Compute the representative matrix for the base using the scaled dataframe
    A = avg_data(index_base, df_scaled).astype(float)
    # Calculate pairwise Euclidean distances between base representatives
    distances = pdist(A, metric='euclidean')
    # Convert the condensed distance matrix to a square form
    distance_matrix = squareform(distances)
    # Extract the upper triangular part of the distance matrix (to avoid duplicate pairs)
    triangular = np.triu(distance_matrix)
    # Flatten the upper triangular matrix to a vector
    vector = triangular.flatten()
    # Sort the vector and remove zeros (self-distances)
    Vec = np.sort(vector[vector > 0])

    W_i = []  # List to store the computed W_i values for each threshold

    # Iterate over each unique distance value in Vec
    for threshold in Vec:
        # Find positions in the triangular matrix where the value is less than or equal to the threshold
        positions = positions_less_or_equal(triangular, threshold)

        # If no such positions, append 0 and continue
        if positions.size == 0:
            W_i.append(0)
            continue

        # Collect all unique indices involved in these positions
        merged_indices = set(positions.flatten())
        merged_indices = list(merged_indices)

        # For each position, merge the corresponding base subsets to form new base elements
        new_base = [np.concatenate((index_base[pos[0]], index_base[pos[1]])) for pos in positions]

        current_base = new_base  # The new base for this threshold

        # If no new base elements, append 0 and continue
        if not current_base:
            W_i.append(0)
            continue

        # Compute the representative matrix for the new base using the original dataframe
        A_for_new_base = avg_data(current_base, df).astype(float)
        w = 0  # Initialize the sum of squared differences

        # For each new base element, compute the sum of squared differences
        for l in range(len(A_for_new_base)):
            indices = current_base[l]
            indices = np.array(indices, dtype=int)
            diff_vectors = A_for_new_base[l] - df.iloc[indices].values  # Difference vectors
            w += np.sum(np.concatenate(np.square(diff_vectors)))    # Sum of squared differences

        W_i.append(w)  # Store the computed value

    r = 0.99  # Regularization parameter for regression
    # Fit a polynomial regression to the W_i values
    poly1, _ = multi_regression(Vec, W_i, r)
    # Compute squared differences between actual and predicted W_i values
    squared_differences = (np.array(W_i) - np.array([evaluate_polynomial(v, poly1)[0] for v in Vec])) ** 2
    # Find the optimal threshold (x_min) that minimizes the squared differences
    x_min = optimal_point(Vec, squared_differences)

    # Find positions in the triangular matrix where the value is less than or equal to x_min
    positions = positions_less_or_equal(triangular, x_min)
    merged_indices = []
    # Collect all unique indices involved in these positions
    for i in range(len(positions)):
        merged_indices += list(positions[i])
    merged_indices = set(merged_indices)
    merged_indices = list(merged_indices)

    new_base = []
    # For each position, merge the corresponding base subsets to form new base elements
    for i in range(len(positions)):
        new_base.append(Base[positions[i][0]] + Base[positions[i][1]])

    # The new gen_base_2 is the union of the original base and the newly formed base elements
    gen_base_2 = Base + new_base

    # Remove base elements that were merged (indices in merged_indices)
    for i in range(len(merged_indices)):
        gen_base_2.remove(Base[merged_indices[i]])

    return gen_base_2

def build_clusters(df):
    """Cluster countries based on their math, reading, and science scores."""
    # Select the columns to use for clustering
    score_columns = ["Mathematics Score", "Reading Score", "Science Score"]
    # Standardize the selected score columns
    scaler = StandardScaler()
    X = df[score_columns]
    X = scaler.fit_transform(X)
    # Set the number of clusters
    N_CLUSTERS = 4
    # Initialize and fit KMeans clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(X)
    # Group indices of countries by their assigned cluster
    clusters = [[str(i) for i in np.where(labels == cluster_id)[0]] for cluster_id in range(N_CLUSTERS)]
    return clusters

# Load the original dataset
df_original = load_data()

# List of qualitative columns to exclude from numerical analysis
Cualitativo = ['Country', 'Type of Economy', 'Population Density', 'Type of Government', 'Continent']

# Remove qualitative columns to retain only quantitative data for analysis
df = df_original.drop(Cualitativo, axis=1)

# Preprocess the quantitative data (e.g., scaling/normalization)
df_scaled = preprocess_data(df)

# Build the initial base (collection of subsets) from the scaled data
Base = build_base(df_scaled)

# Build build_generalized_base_1 using the initial base and the unscaled quantitative data
gen_base_1 = build_gen_base_1(Base, df)

# Build gen_base_2 using the initial base, scaled data, and unscaled data
gen_base_2 = build_gen_base_2(Base, df_scaled, df)

# Cluster the countries based on their scores and obtain cluster groupings
clusters = build_clusters(df)


def define_groups_and_targets(df, gen_base_1, gen_base_2, Base, clusters):
    """Define groupings and target sets for analysis based on qualitative columns (translated to English)."""

    # Economy type groups
    HighEconomy = list(df[df['Type of Economy'] == 'High'].index.astype(str))
    MediumHighEconomy = list(df[df['Type of Economy'] == 'Medium-high'].index.astype(str))
    MediumLowEconomy = list(df[df['Type of Economy'] == 'Medium-low'].index.astype(str))

    # Population density groups
    LowDensity = list(df[df['Population Density'] == 'Low'].index.astype(str))
    HighDensity = list(df[df['Population Density'] == 'High'].index.astype(str))
    MediumDensity = list(df[df['Population Density'] == 'Medium'].index.astype(str))
    ExtremeDensity = list(df[df['Population Density'] == 'Extreme'].index.astype(str))

    # Government type groups
    PoorDemocracy = list(df[df['Type of Government'] == 'Poor democracy'].index.astype(str))
    FullDemocracy = list(df[df['Type of Government'] == 'Full democracy'].index.astype(str))
    Authoritarianism = list(df[df['Type of Government'] == 'Authoritarianism'].index.astype(str))
    HybridRegime = list(df[df['Type of Government'] == 'Hybrid regime'].index.astype(str))

    # Continent groups
    Europe = list(df[df['Continent'] == 'Europe'].index.astype(str))
    SouthAmerica = list(df[df['Continent'] == 'South America'].index.astype(str))
    Oceania = list(df[df['Continent'] == 'Oceania'].index.astype(str))
    Asia = list(df[df['Continent'] == 'Asia'].index.astype(str))
    NorthAmerica = list(df[df['Continent'] == 'North America'].index.astype(str))
    Africa = list(df[df['Continent'] == 'Africa'].index.astype(str))

    # List of all indices as strings
    A = [str(i) for i in range(len(df))]

    # Dictionary of target groups for closure/interior analysis
    target_groups = {
        "gen_base_1": gen_base_1,
        "gen_base_2": gen_base_2,
        "base": Base,
        "clusters": clusters,
    }

    # Group definitions for economy types
    group_definitions_economy = {
        "HighEconomy": HighEconomy,
        "MediumHighEconomy": MediumHighEconomy,
        "MediumLowEconomy": MediumLowEconomy,
    }

    # Group definitions for population density
    group_definitions_density = {
        "LowDensity": LowDensity,
        "HighDensity": HighDensity,
        "MediumDensity": MediumDensity,
        "ExtremeDensity": ExtremeDensity,
    }

    # Group definitions for government types
    group_definitions_government = {
        "PoorDemocracy": PoorDemocracy,
        "FullDemocracy": FullDemocracy,
        "Authoritarianism": Authoritarianism,
        "HybridRegime": HybridRegime,
    }

    # Group definitions for continents
    group_definitions_continent = {
        "Europe": Europe,
        "SouthAmerica": SouthAmerica,
        "Oceania": Oceania,
        "Asia": Asia,
        "NorthAmerica": NorthAmerica,
        "Africa": Africa,
    }

    # Return all groupings and targets
    return (
        A,
        target_groups,
        group_definitions_economy,
        group_definitions_density,
        group_definitions_government,
        group_definitions_continent
    )

# Define all groupings and targets for analysis
(
    A,
    target_groups,
    group_definitions_economy,
    group_definitions_density,
    group_definitions_government,
    group_definitions_continent
) = define_groups_and_targets(
    df_original,
    gen_base_1,
    gen_base_2,
    Base,
    clusters
)

# Process and output results for each group type
process_batch(group_definitions_economy, df_original, target_groups, A, output_folder="results/economy")      # Economy-based groups
process_batch(group_definitions_density, df_original, target_groups, A, output_folder="results/density")      # Population density-based groups
process_batch(group_definitions_government, df_original, target_groups, A, output_folder="results/government")# Government type-based groups
process_batch(group_definitions_continent, df_original, target_groups, A, output_folder="results/continent")  # Continent-based groups