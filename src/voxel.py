import numpy as np
import pandas as pd
import SimpleITK as sitk
"""
Class to create voxel object from .nii.gz files

Inputs file names (given with full directory location) for 
average_template, annotation_boundary, and annotation_boundary.

ordering ='xyz' or 'zyx' specifies whether the indices of image arrays 
go from x-direction to z, or z to x respectively. The object standardizes 
the arrays so that the arrays are always stored in 'zyx' format.
"""
class voxel:
    def __init__(self, 
                 file,
                 ordering = 'zyx'):
        self.ordering = ordering
        self.data = self.readfile(file)
        self.Lx, self.Ly, self.Lz = self.GetSize() #extents of the coordinate system
        self.dx, self.dy, self.dz = self.GetSpacing() #spacings
        
    def readfile(self, file_name):
        image = sitk.ReadImage(file_name)
        array = sitk.GetArrayViewFromImage(image).copy()
        if self.ordering == 'xyz': # change xyz ordering to zyx
            array = np.moveaxis(array, [0,1,2], [2,1,0])
        return {'image': image, 'array': array}

    def GetSize(self):
        size = list(self.data['image'].GetSize())
        if self.ordering == 'xyz':
            size = size[::-1]
        return size

    def GetSpacing(self):
        return self.data['image'].GetSpacing()

    def image_info(self):
        print(' size: ' + str(self.GetSize()) + ' voxels')
        print(' spacing: ' + str(self.GetSpacing()) + ' mm' )
