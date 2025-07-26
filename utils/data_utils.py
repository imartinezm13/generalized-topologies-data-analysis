import numpy as np

def stand(data):
    """
    Standardizes the input DataFrame by scaling each column to the range [0, 1].

    Parameters:
    - data: A pandas DataFrame containing the data to be standardized.

    Returns:
    - A numpy array with the scaled values.
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_max = np.where(data_max - data_min == 0, 1, data_max)
    return (data - data_min) / (data_max - data_min)

def positions_less_or_equal(matrix, number):
    """
    Returns the positions of elements in the matrix that are less than or equal to the given number and greater than 0.

    Parameters:
    - matrix: A numpy array or 2D matrix.
    - number: The threshold number.

    Returns:
    - A numpy array of the positions (indices) where the condition holds.
    """
    return np.argwhere((matrix <= number) & (matrix > 0))

def avg_data(Base, data):
    """
    Computes the average (mean) value of elements in data corresponding to the indices in Base.

    Parameters:
    - Base: List of lists with numerical indices.
    - data: A pandas Series or DataFrame containing the data to compute the mean.

    Returns:
    - A list of means for each group in Base.
    """
    data_array = data.to_numpy()
    averages = np.array([data_array[group].mean(axis=0) for group in Base])
    return averages
