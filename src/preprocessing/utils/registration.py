import nibabel as nib
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.io.utils import create_nifti_header

from nipype.interfaces import fsl
from tqdm import tqdm
import logging
import os

class Registration():
    def __init__(self, name_nii, atlas_path, output_dir, name, img_type):
        self.name_nii = name_nii
        self.atlas = atlas_path
        self.name = name
        self.img_type = img_type
        self.output_dir = output_dir

    def run(self):
        fl = fsl.FLIRT()
        fl.inputs.in_file = self.name_nii
        fl.inputs.reference = self.atlas
        fl.inputs.out_file = self.output_dir + self.name + '_reg.nii.gz'
        fl.inputs.output_type = 'NIFTI_GZ'
        fl.inputs.bins = 1500
        #fl.inputs.rigid2D = True
        fl.inputs.dof = 12
        # For 3D to 3D mode the DOF can be set to 12 (affine), 9 (traditional), 7 (global rescale) or 6 (rigid body)
        fl.inputs.out_matrix_file = self.output_dir + self.name + '_reg_matrix.mat'
        
        # PET images use directly the matrix previously computed registering the binary mask
        if self.img_type != 'pet':
            # Produce the registration matrix on the first volume of the image...
            try:
                out_fl = fl.run()
            except Exception as e:
                logging.error(e)
                logging.error(f"{self.name_nii} at Registration.run()")
                print(e)
                print(f"{self.name_nii} at Registration.run()")
        
        # ... and use the matrix to register all the volumes
        fl.inputs.apply_xfm = True
        fl.inputs.in_matrix_file = self.output_dir + self.name + '_reg_matrix.mat'
        
        try:
            out_fl = fl.run()
        except Exception as e:
            logging.error(e)
        
        registered = out_fl.outputs.out_file
        img = nib.load(registered)
        
        return img.get_fdata(), img.affine, img.header