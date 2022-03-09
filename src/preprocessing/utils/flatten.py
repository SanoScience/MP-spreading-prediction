from nipype.interfaces.fsl.maths import MeanImage
from nibabel import load
import os
import logging

class Flatten:
    name_nii = None
    atlas = None
    final_out_name = None
    name = None
    
    def __init__(self, name_nii, name):
        self.name_nii = name_nii
        self.name = name
    
    def run(self):
        # Flatten 4D image in 3D mean image across Time
        mi = MeanImage()
        mi.inputs.in_file = self.name_nii
        mi.inputs.dimension = 'T'
        mi.inputs.nan2zeros = True
        mi.inputs.out_file = self.name + '.nii.gz'
        mi.inputs.output_type = 'NIFTI_GZ'

        try:
            out_mi = mi.run()
        except Exception as e:
            logging.error(e)

        flattened_path = out_mi.outputs.out_file
        img = load(flattened_path)
        return img.get_fdata(), img.affine, img.header