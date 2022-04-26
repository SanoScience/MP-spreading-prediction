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

    def __init__(self, path_file, name, img_type):
        """
        This object demands a path file, not the data. 
        Be aware that this object is a wrapper for the fsl toolkit, meaning that you need to have
        fsl in your machine: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
        """
        self.name = name
        self.path_file = path_file
        self.img_type = img_type

    def run(self, frac=.3):
        # Nipype wrapping of BET is skipped due to instability
        
        if self.img_type == 'dwi': # multiple volumes images are truncated to the first volume to do the brain extraction
            input_be = 'first_slice.nii.gz'
            img = load(self.path_file)
            data, affine, header = img.get_fdata()[:,:,:,0], img.affine, img.header
            save(Nifti1Image(data, affine, header), input_be)
        else:
            input_be = self.path_file
        
        # Two cuts: the first with a very negative vertical gradient to cut the neck, the second more balanced to extract all the brain
        if self.img_type == 'anat':
            os.system(f"bet2 {self.path_file} {self.name} -f {frac} -g -1")
            self.path_file = self.name
        
        os.system(f"bet2 {input_be} {self.name} -m -f {frac} -g 0")
        self.binary_mask = self.name + '_mask.nii.gz'

        if self.img_type == 'dwi':        
            os.system(f"fslmaths {self.path_file} -mas {self.binary_mask} {self.name}.nii.gz")
            os.system(f"rm {input_be}")

        img = load(self.name+'.nii.gz')
        return img.get_fdata(), img.affine, img.header

    def get_mask(self):
        img = load(self.binary_mask)
        return img.get_fdata()