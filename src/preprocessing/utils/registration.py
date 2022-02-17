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
    """
    ==========================================
    Registration of DWI volumes to an atlas
    ==========================================
    """
    data = None
    header = None 
    affine = None
    bvals = None
    bvecs = None
    gtab = None
    name = None
    
    def __init__(self, data, affine, header, name):
        self.data = data
        self.header = header
        self.affine = affine
        self.name = name

    def affine_reg(self, static_img, static_affine, moving_img, moving_affine):
        """
        Implements an affine registration between just two images.
        static_affine and moving_affine are always self.affine ONLY in motion correction (in atlas reg. they are not)
        """
        c_of_mass = transform_centers_of_mass(static_img,
                                        static_affine,
                                        moving_img,
                                        moving_affine)

        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)

        #level_iters = [10000, 1000, 100]
        level_iters = [1000, 100, 10]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        affreg = AffineRegistration(metric=metric,
                                    level_iters=level_iters,
                                    sigmas=sigmas,
                                    factors=factors,
                                    verbosity=0)

        transform = TranslationTransform3D()
        params0 = None
        starting_affine = c_of_mass.affine
        translation = affreg.optimize(static_img, moving_img, transform, params0,
                                    static_affine, moving_affine,
                                    starting_affine=starting_affine)
        transformed = translation.transform(moving_img)
        self.T1 = c_of_mass.affine

        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static_img, moving_img, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=starting_affine)
        transformed = rigid.transform(moving_img)
        self.T2 = translation.affine

        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        align = affreg.optimize(static_img, moving_img, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=starting_affine)
        transformed = align.transform(moving_img)
        self.T3 = rigid.affine

        # TODO: apply transformations to the same image subsequentially and compare it to 'transformed'
        return transformed, align.affine, align

    def run(self, atlas, gtab=None):
        """
        Registers the first volume of an image to the atlas. 
        It assumes that all the other volumes are all correctly aligned with the first volume.
        """
        
        # Loading the atlas
        atlas_img = nib.load(atlas)
        atlas_data = atlas_img.get_fdata()
        self.atlas_affine = atlas_img.affine

        ### Anatomical iamge ###
        if len(self.data.shape)==3: 
            self.data_registered, self.regis_affine, regis = self.affine_reg(
                                            atlas_data, self.atlas_affine,
                                            self.data, self.affine
                                            )
        ### dwi image ###
        else:
            # Regsitering the first volume b0 to the atlas
            #self.data_registered = np.zeros(self.data.shape)
            first_volume, self.regis_affine, regis = self.affine_reg(
                                                atlas_data, self.atlas_affine,
                                                self.data[...,0], self.affine
                                                )
            # Now we apply the transformation to all other volumes of the image.
            self.data_registered = np.zeros(
                (first_volume.shape[0], first_volume.shape[1], first_volume.shape[2], self.data.shape[3])
                )
            self.data_registered[..., 0] = first_volume
            volumes_affines = []
            #volumes_affines.append(self.regis_affine)
            for i in range(1, self.data.shape[-1]):
                self.data_registered[...,i] = regis.transform(self.data[...,i])
                if self.gtab and not gtab.b0s_mask[i]:
                    volumes_affines.append(self.regis_affine)

            # We apply the B-Matrix correction (we have moved the DWI image!)
            if self.gtab:
                self.gtab_corrected = reorient_bvecs(gtab, volumes_affines)
                np.savetxt(self.name + '.bval', self.bvals)
                np.savetxt(self.name + '.bvec', self.gtab_corrected.bvecs)
        
        return self.data_registered, self.atlas_affine, self.header 

    def get_transformation(self):
        return self.regis_affine, self.T3, self.T2, self.T1
                                
class RegistrationPET():
    def __init__(self, name_nii, atlas_path, name, img_type='dwi'):
        self.name_nii = name_nii
        self.atlas = atlas_path
        self.name = name
        self.img_type = img_type

    def run(self):
        fl = fsl.FLIRT(bins=1080, cost_func='mutualinfo')
        fl.inputs.in_file = self.name_nii
        fl.inputs.reference = self.atlas
        fl.inputs.cost = 'mutualinfo'
        fl.inputs.out_file = self.name + '.nii'
        fl.inputs.output_type = 'NIFTI'
        fl.inputs.out_matrix_file = self.name + '_reg_matrix.mat'
            
        # for dwi, create the matrix and apply it to all volumes. Masks use the already existing registration matrix
        if self.img_type == 'dwi':
            '''
            echo "creating matrix"
            flirt -ref $atlas -in $flirt_input -cost mutualinfo -searchcost mutualinfo -omat $mat

            echo "atlas registration"
            flirt -ref $atlas -in $flirt_input -applyxfm -init $mat -out $output 
            '''
            try:
                out_fl = fl.run()
            except Exception as e:
                logging.error(e)
                
        if self.img_type == 'dwi' or self.img_type == 'mask':
            fl.inputs.apply_xfm = True
            fl.inputs.in_matrix_file = self.name + '_reg_matrix.mat'   
        
        if self.img_type == 'pet':
            fl.inputs.no_resample = True

        try:
            out_fl = fl.run()
        except Exception as e:
            logging.error(e)

        registered = out_fl.outputs.out_file
        img = nib.load(registered)
        
        return img.get_fdata(), img.affine, img.header