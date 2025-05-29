from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


class transform_ccf_to_reconstructed:
    """
    Class to create a function that converts CCF coordinates to 
    MERFISH reconstructed coordinates.

    Input:

        df_cell_xy: 
	    Pandas dataframe object containing MERFISH cell coordinate
	    details. At minimum, the columns of the dataframe be
	    [subclass, x_ccf, y_ccf, z_ccf, x_reconstructed,
	    y_reconstructed, z_reconstructed,
	    deriv_11,deriv_12,deriv_21,deriv_22] The function below
	    will help in creating the columns of derivatives deriv_ij
	    from the Allen brain merfish dataframes. The df_cell_xy
	    is assumed to be a dataframe that has undergone that
	    derivative finding step.

        num_neighbors:
        Number of nearest neighbor MERFISH cells to location 
        coord_ccf that are used in averaging of the gradient
        to perform the CCF to MERFISH transformation.
        num_neighbors=2 is found to be good.
    """
    def __init__(self, df_cell_xy, num_neighbors=2):
        self.df_cell_xy = df_cell_xy
        self.num_neighbors = num_neighbors
        # initialize sklearn nearest neighbor using merfish dataset
        k_neighbors = 3
        ref_list = self.df_cell_xy[['z_ccf', 'y_ccf', 'x_ccf' ]].to_numpy()
        self.nbrs = NearestNeighbors(n_neighbors=1 + k_neighbors, algorithm='ball_tree').fit(ref_list)

    def get_reconstructed(self, coord_ccf):
        """
        Input: coord_ccf
	    np.array of shape (B, D=3) where B is the batchsize,
	    and D is the three CCF coordinates.

        Output: (B, D) numpy array containing estimated reconstructed coordinates.
        """
        coord_reconstructed = np.zeros_like(coord_ccf, dtype=np.float32)
        distances, indices = self.nbrs.kneighbors(coord_ccf)
        for i_neighbors in range(self.num_neighbors):
            cell_nn = self.df_cell_xy.iloc[indices.T[i_neighbors]]
            F_nn = cell_nn[['x_reconstructed','y_reconstructed', 'z_reconstructed']].to_numpy().T
            x_nn = cell_nn[['z_ccf', 'y_ccf']].to_numpy().T
            dF11, dF12, dF21, dF22 = cell_nn[['deriv_11', 'deriv_12', 'deriv_21', 'deriv_22']].to_numpy().T
            coord_reconstructed.T[0] = coord_reconstructed.T[0] + F_nn[0] +\
                 dF11*(coord_ccf.T[0]-x_nn[0]) +\
                 dF12*(coord_ccf.T[1]-x_nn[1])
        
            coord_reconstructed.T[1] = coord_reconstructed.T[1] + F_nn[1] +\
                 dF21*(coord_ccf.T[0]-x_nn[0]) +\
                 dF22*(coord_ccf.T[1]-x_nn[1])
        
            if i_neighbors == 1:
                coord_reconstructed.T[2] = F_nn[2]
     
        coord_reconstructed.T[:2] = coord_reconstructed.T[:2]/self.num_neighbors
        return coord_reconstructed



def precompute_derivatives(df_allen_cell, 
                           parcellation_structure = 'VISp', 
                           output = None):
    """ 
    Function to compute derivatives deriv_ij:

        deriv_ij = d(i-th reconstructed coord component)/d(j-th CCF coord component),

    at MERFISH cell locations, and prepare a new pruned dataframe
    containing celllabels, subclass, MERFISH coord, CCF coord, and
    derivatives deriv_11, deriv_12, deriv_21, deriv_22.

    Input:
	df_allen_cell: Allen brain cell data with all parcellation
	info included parcellation_structure: anatomical region to
	focus on.  output = file name to store this dataframe, if
	needed.

    Output:
    Prepared dataframe with columns: ['subclass', 'x_ccf',
    'y_ccf', 'z_ccf', 'x_reconstructed', 'y_reconstructed',
    'z_reconstructed'].  Also, saves the prepared dataframe if
    output is not None.
    """

    select = (df_allen_cell['parcellation_structure'] == 'VISp')
    cell_xy = df_allen_cell[select][['subclass', 'x_ccf', 'y_ccf', 'z_ccf', 
                                     'x_reconstructed', 'y_reconstructed', 'z_reconstructed']]
    #
    # Use sklearn NearestNeighbors to get the k_neighbor nearest neighbors to a 
    # MERFISH cell.
    k_neighbors = 6
    ref_list = cell_xy[['z_ccf', 'y_ccf', 'x_ccf' ]].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=1 + k_neighbors, algorithm='ball_tree').fit(ref_list)
    _distances, _indices = nbrs.kneighbors(cell_xy[['z_ccf', 'y_ccf', 'x_ccf' ]].to_numpy())
    cell_xy_nn = [cell_xy.iloc[_indices.T[k+1]] for k in range(k_neighbors)]
    #
    # Let F be MERFISH reconstruced coordinates.
    # Let x be CCF coordinates.
    # Assuming nearest neighbor MERFISH cell 'c' is close enough, to first order
    # in distance, we can express F at an arbitrary position close to 'c' as 
    #    F_i(p) = F_i(c) + \sum_j dF_i/dx_j(c) * ( x_j(p) - x_j(c) )
    #
    # To do the above, we need the derivatives dF_i/dx_j(c). For this,
    # we use 4 MERFISH cells where F_i and x_j are all known, we solve for 
    # dF_i/dx_j(c). For each cell, we repeat this for many such nearest neighbor
    # pairings and find the mean as a better estimate.  We also clip the resulting 
    # derivative to be ~[-1,1].
    #
    F_fixed = cell_xy[['x_reconstructed','y_reconstructed'] ].to_numpy()
    x_fixed = cell_xy[['z_ccf', 'y_ccf']].to_numpy()

    F_fixed_nn = []
    x_fixed_nn = []
    for nn in cell_xy_nn:
        F_fixed_nn.append(nn[['x_reconstructed','y_reconstructed']].to_numpy())
        x_fixed_nn.append(nn[['z_ccf', 'y_ccf']].to_numpy())

    x_fixed_nn = np.array(x_fixed_nn)
    F_fixed_nn = np.array(F_fixed_nn)
    x_fixed = np.array([x_fixed for _ in range(k_neighbors)])
    F_fixed = np.array([F_fixed for _ in range(k_neighbors)])

    dx = x_fixed_nn - x_fixed
    dF = F_fixed_nn - F_fixed

    # dx[n,m,nu] -> dx[m,n,nu]
    dx = dx.transpose((1, 0, 2))
    # F[n,m,nu] -> F[nu, m, n]
    dF = dF.transpose((1, 0, 2))

    c, n, m = dx.shape
    dx = dx.reshape(c,n//2,2,m)
    dF = dF.reshape(c,n//2,2,m)

    dxrec_by_dxccf = np.linalg.solve(dx, dF)
    dxrec_by_dxccf = np.clip(dxrec_by_dxccf, -1.0, 1.0)
    dxrec_by_dxccf = dxrec_by_dxccf.transpose((3,2,1,0))

    cell_xy['deriv_11'] = np.mean(dxrec_by_dxccf[0,0,:,:], axis=0)
    cell_xy['deriv_12'] = np.mean(dxrec_by_dxccf[0,1,:,:], axis=0)
    cell_xy['deriv_21'] = np.mean(dxrec_by_dxccf[1,0,:,:], axis=0)
    cell_xy['deriv_22'] = np.mean(dxrec_by_dxccf[1,1,:,:], axis=0)

    if output is not None:
        cell_xy.to_csv(output)
    return cell_xy
