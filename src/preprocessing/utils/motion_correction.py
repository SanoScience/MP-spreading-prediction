import multiprocessing
from time import time, sleep
import nibabel as nib
import numpy as np
from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

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
        
    def affine_reg(self, static_img, static_affine, moving_img, moving_affine, index, queue):
        """
        Implements an affine registration between just two images.
        static_affine and moving_affine are always self.affine ONLY in motion correction (in atlas reg. they are not)
        """
        try:
            c_of_mass = transform_centers_of_mass(static_img,
                                            static_affine,
                                            moving_img,
                                            moving_affine)

            nbins = 32
            sampling_prop = None
            metric = MutualInformationMetric(nbins, sampling_prop)

            level_iters = [300, 100, 10]
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
            del c_of_mass
            translation = affreg.optimize(static_img, moving_img, transform, params0,
                                        static_affine, moving_affine,
                                        starting_affine=starting_affine)
            transformed = translation.transform(moving_img)

            transform = RigidTransform3D()
            params0 = None
            starting_affine = translation.affine
            del translation
            rigid = affreg.optimize(static_img, moving_img, transform, params0,
                                    static_affine, moving_affine,
                                    starting_affine=starting_affine)
            transformed = rigid.transform(moving_img)

            transform = AffineTransform3D()
            params0 = None
            starting_affine = rigid.affine
            del rigid
            align = affreg.optimize(static_img, moving_img, transform, params0,
                                    static_affine, moving_affine,
                                    starting_affine=starting_affine)
            transformed = align.transform(moving_img)
        except Exception as e:
            logging.error(e)
            queue.put_nowait((index, moving_img))
        else:
            queue.put_nowait((index, transformed))
            
        return 
        
    def run(self):
        """
        Registering all the volumes of the image with the first b0 volume. 
        """

        static_img = self.data[...,0]
        self.corrected_data[...,0] = static_img

        q = multiprocessing.Queue()
        procs = []
        for i in range(1, self.data.shape[-1]):
            # We do not need to correct the first volume to itself
            
            p = multiprocessing.Process(target=self.affine_reg, args=(static_img, self.affine, self.data[...,i], self.affine, i, q))
            p.start()
            procs.append(p)
        
        counter = 0            
        while counter < self.data.shape[-1]-1:
            index, volume = q.get()
            self.corrected_data[...,index] = volume
            counter += 1

        assert self.corrected_data.shape[-1] == self.data.shape[-1]
        return self.corrected_data, self.affine, self.header 