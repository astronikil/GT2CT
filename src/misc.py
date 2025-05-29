import numpy as np
from .utils import crop_structure
from .sampling import sample_from_probability_array

def dist_from_ccf_to_nearest_merfish(annotation_ccf_voxelobj, 
                                     parcels_in_struc,
                                     zindex, 
                                     func_ccfdistance_to_nearest_merfish,
                                     side = 'right'):
    im, im_range = crop_structure(vox = annotation_ccf_voxelobj, 
                                  zindex = zindex, 
                                  parcels_in_struc = parcels_in_struc, 
                                  annotation_voxobj = annotation_ccf_voxelobj)
    arr = im[side].copy()
    select = np.where(arr > 0)
    arr[select] = 1.0
    p_mat = arr/arr.sum()
    y, x = sample_from_probability_array(p_mat, num_samples=200)
    y_coord = (y + im_range[side]['min_row'])
    x_coord = (x + im_range[side]['min_col']) 
    if side == 'right':
        x_coord = x_coord + annotation_ccf_voxelobj.Lx//2 
    z_list = np.ones_like(x_coord, dtype=np.float32)*zindex
    r_phys = np.array([x_coord * annotation_ccf_voxelobj.dx, 
                       y_coord * annotation_ccf_voxelobj.dy, 
                       z_list * annotation_ccf_voxelobj.dz]
                      ).T
    distances = func_ccfdistance_to_nearest_merfish(r_phys)
    return zindex*1.0, distances.flatten().mean(), distances.flatten().std()
