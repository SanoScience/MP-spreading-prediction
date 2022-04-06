from nibabel import save, Nifti1Image, load
from dipy.segment.mask import median_otsu
from nipype.interfaces import fsl
import numpy as np
import logging
import os

class BrainExtraction:
    data = None
    header = None 
    affine = None
    mask = None
    binary_mask = None
    name = None

    def __init__(self, data, affine, header, name):
        self.data = data
        self.header = header
        self.affine = affine
        self.name = name

    def run(self, numpass = 4, median_radius = 4, dilate = 1):
        if len(self.data.shape)==4:
            vol_idx = range(len(self.data[0,0,0,:]))
        else:
            vol_idx = None
        self.mask, self.binary_mask = median_otsu(self.data, vol_idx=vol_idx, numpass=numpass, median_radius=median_radius, dilate=dilate) #Note: autocrop is suggested only for DWI images with voxel resolution upsampled to 1x1x1
        return self.mask, self.affine, self.header
    
    def get_mask(self):
        return self.binary_mask

class BET_FSL:

    def __init__(self, path_file, name):
        """
        This object demands a path file, not the data. 
        Be aware that this object is a wrapper for the fsl toolkit, meaning that you need to have
        fsl in your machine: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
        """
        self.name = name
        self.path_file = path_file

    def run(self, frac=.4, vertical_gradient=-.5):
        # Nipype wrapping of BET is skipped due to instability
        os.system(f"bet2 {self.path_file} {self.name} -m -f {frac} -g {vertical_gradient}")

        self.binary_mask = self.name + '_mask.nii.gz'
        img = load(self.name + '.nii.gz')
        return img.get_fdata(), img.affine, img.header

    def get_mask(self):
        img = load(self.binary_mask)
        return img.get_fdata()