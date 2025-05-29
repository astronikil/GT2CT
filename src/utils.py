import numpy as np

def crop_structure(vox, 
                   zindex, 
                   parcels_in_struc, 
                   annotation_voxobj):
    """
    Crops numpy array of 2D voxels from a z-slice to the smallest square
    that encloses an brain structure -- VISp in this specific case

    Args:
        vox:  voxel object
        zindex: integer specifying the z-slice

    Returns:
        two dictionaries with keys 'left' and 'right'
        first dict contains the cropped arrays in left and right half of brain
        second dict contains the location of the cropped array in the larger array
    """
    x_mid = vox.Lx//2

    annotate_slice = annotation_voxobj.data['array'][zindex, :, :].copy()
    index = np.isin(annotate_slice, parcels_in_struc)

    arr_in = vox.data['array'][zindex, :, :].copy()

    arr_in = np.clip(arr_in, a_min=0, a_max=None)

    arr_in[~index] = 0  #Set non-visp elements to 0

    crop_left = arr_in[:,:x_mid]
    crop_right = arr_in[:,x_mid:]

    crop_right, arr_range_right = get_smallest_square_containing_nonzero(crop_right)
    crop_left, arr_range_left = get_smallest_square_containing_nonzero(crop_left)

    return {'left': crop_left, 'right': crop_right},\
           {'left': arr_range_left, 'right': arr_range_right}

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


def apply_threshold_matrix(arr, low_thresh = 0, high_thresh = 200000):
    assert high_thresh > low_thresh, 'high_thresh should be greater than low_thresh'
    p_array = arr.astype(np.float32).copy() - low_thresh
    new_high_thresh = high_thresh - low_thresh
    p_array[p_array < 0] = 0 #Set -ve values to zero
    p_array[p_array > new_high_thresh] = new_high_thresh
    return p_array
