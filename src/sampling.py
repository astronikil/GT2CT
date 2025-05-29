import numpy as np
from sklearn.neighbors import NearestNeighbors

def sample_from_probability_array(prob_array, num_samples=1):
    """
    The input array should contain positive entries that lie between 
    0 and 1.  Sum of all entries of prob_array should be 1.

    With the above restriction, each entry of input array is the probability of 
    that array index to be picked in a random sampling.

    This function performs such random sampling of indices from the input array.

    Args:
        prob_array (np.ndarray): A NxN NumPy array where each entry represents
                                  the probability of that element being chosen.
                                  The sum of all entries should be 1.
        num_samples (int): The number of (row,col) samples to draw.

    Returns:
        tuple: A tuple of two NumPy arrays, `rows` and `cols`, containing the row
               and column indices of the sampled elements. Each array will have
               a length equal to `num_samples`.
    """
    flat_probabilities = prob_array.flatten()
    indices = np.arange(flat_probabilities.size)
    sampled_indices = np.random.choice(indices, size=num_samples, p=flat_probabilities)
    rows = np.floor_divide(sampled_indices, prob_array.shape[1])
    cols = np.mod(sampled_indices, prob_array.shape[1])
    return rows, cols

def create_frequency_matrix_vectorized(rows, cols, n):
    """
    Creates an NxN matrix where each element represents the frequency of
    the corresponding index being sampled, using vectorization.

    Args:
        rows (np.ndarray): Array of row indices of the sampled elements.
        cols (np.ndarray): Array of column indices of the sampled elements.
        n (int): The dimension of the original NxN probability array.

    Returns:
        np.ndarray: An NxN matrix of frequencies.
    """
    index = np.ravel_multi_index((rows, cols), dims=(n, n))
    frequency_array = np.bincount(index, minlength=n*n).reshape(n, n)
    return (frequency_array*1.0)/(frequency_array.sum() + 1.0E-12)



def get_smallest_square_containing_nonzero(A):
    """
    Finds the smallest square matrix containing all non-zero values of A.

    Args:
        A: A 2D numpy array.

    Returns:
        A square numpy array containing all non-zero values of A,
        or None if A is all zeros.
    """
    # Find the indices of the non-zero elements
    nonzero_indices = np.nonzero(A)
    if nonzero_indices[0].size == 0:
        return None  # Handle the case where A is all zeros

    # Find the min and max row and column indices
    min_row = np.min(nonzero_indices[0])
    max_row = np.max(nonzero_indices[0])
    min_col = np.min(nonzero_indices[1])
    max_col = np.max(nonzero_indices[1])

    # Calculate the side length of the square
    side_length = max(max_row - min_row + 1, max_col - min_col + 1)

    # Adjust max_row and max_col if necessary to make the matrix square
    max_row = max(max_row, min_row + side_length - 1)
    max_col = max(max_col, min_col + side_length - 1)

    # Extract the square submatrix
    square_matrix = A[min_row:max_row + 1, min_col:max_col + 1]
    return square_matrix, {'min_row': min_row, 'max_row': max_row, 
                           'min_col': min_col, 'max_col': max_col}

def sample_indices_from_probability_array(A: np.ndarray, N_sample: int) -> np.ndarray:
    """
    Samples N_sample number of indices from a multi-dimensional NumPy array A
    according to their probabilities.

    Args:
        A (np.ndarray): A multi-dimensional NumPy array where each element
                        is between 0 and 1, and the sum of all elements is 1.
                        Each element represents the probability that its index is picked.
        N_sample (int): The number of indices to sample.

    Returns:
        np.ndarray: A 2D NumPy array of shape (N_sample, A.ndim), where each row
                    represents a sampled index (e.g., [l, n, m] for a 3D array).
    """
    if not np.isclose(A.sum(), 1.0):
        print(A.sum())
        raise ValueError("The sum of all elements in array A must be 1.0.")
    if not (A >= 0).all() and (A <= 1).all():
        raise ValueError("All elements in array A must be between 0 and 1.")
    # Flatten the array to treat it as a 1D probability distribution
    flattened_A = A.ravel()
    # Create an array of all possible 1D indices
    # We use arange and then convert these 1D indices back to multi-dimensional indices later
    linear_indices = np.arange(flattened_A.size)

    # Sample linear indices based on their probabilities
    sampled_linear_indices = np.random.choice(
        linear_indices,
        size=N_sample,
        p=flattened_A,
        replace=True  # Usually sampling with replacement for probabilities
    )

    # Convert the sampled linear indices back to multi-dimensional indices
    # using np.unravel_index
    sampled_multi_dim_indices = np.unravel_index(sampled_linear_indices, A.shape)

    # np.unravel_index returns a tuple of arrays, where each array corresponds
    # to a dimension. We need to stack them to get (N_sample, A.ndim)
    return np.column_stack(sampled_multi_dim_indices)

def get_frequency_array(n, m, rows, cols):
     index = np.ravel_multi_index((rows, cols), dims=(n,m))
     frequency_array = np.bincount(index, minlength=n*m).reshape(n, m)
     frequency_array = frequency_array*1.0/(frequency_array.sum() + 1.0E-12)
     return frequency_array
