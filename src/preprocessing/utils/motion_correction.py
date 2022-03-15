import queue
import nibabel as nib
import numpy as np
""" from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data """
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

from multiprocessing import Queue, Process
import logging

class MotionCorrection:
    """
    ==============================
    Motion correction of PET data
    ==============================
    """
    data = None
    header = None 
    affine = None
    corrected_data = None
    name = None
    
    def __init__(self, data, affine, header, name):
        self.data = data
        self.header = header
        self.affine = affine
        self.corrected_data = np.zeros(self.data.shape)
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

        level_iters = [500, 100, 10]
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

        transform = RigidTransform3D()
        params0 = None
        starting_affine = translation.affine
        rigid = affreg.optimize(static_img, moving_img, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=starting_affine)
        transformed = rigid.transform(moving_img)

        transform = AffineTransform3D()
        params0 = None
        starting_affine = rigid.affine
        align = affreg.optimize(static_img, moving_img, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=starting_affine)
        transformed = align.transform(moving_img)
        
        # TODO: apply transformations to the same image subsequentially and compare it to 'transformed'
        return transformed
        
    def run(self):
        """
        Registering all the volumes of the image with the first b0 volume. 
        """

        static_img = self.data[...,0]
        self.corrected_data[...,0] = static_img

        for i in range(1, self.data.shape[-1]):
            # We do not need to correct the first volume to itself
            try:
                transformed = self.affine_reg(static_img, self.affine, self.data[...,i], self.affine)
            except Exception as e:
                logging.error(e)
                logging.error(self.name)
                transformed = self.data[...,i]
            self.corrected_data[...,i] = transformed

        return self.corrected_data, self.affine, self.header 