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

    def run(self, numpass = 5, median_radius = 5, dilate = 1):
        if len(self.data.shape)==4:
            vol_idx = range(len(self.data[0,0,0,:]))
        else:
            vol_idx = None
        self.mask, self.binary_mask = median_otsu(self.data, vol_idx=vol_idx, numpass=numpass, median_radius=median_radius, dilate=dilate)
        return self.mask, self.affine, self.header
    
    def get_mask(self):
        return self.binary_mask

class BET_FSL:
    path_file = None

    def __init__(self, path_file, name, binary_mask=True):
        """
        This object demands a path file, not the data. 
        Be aware that this object is a wrapper for the fsl toolkit, meaning that you need to have
        fsl in your machine: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL
        """
        self.name = name
        img = load(path_file)
        self.affine = img.affine
        self.header = img.header
        self.path_file = path_file
        self.binary_mask = binary_mask

    def run(self, frac=.1, vertical_gradient=-.5, output_type='NIFTI'):
        bet = fsl.BET()
        bet.inputs.in_file = self.path_file
        bet.inputs.frac = frac
        bet.inputs.vertical_gradient = vertical_gradient
        bet.inputs.output_type = output_type
        bet.inputs.mask = self.binary_mask
        bet.inputs.out_file = self.name + '.nii'
        bet.inputs.reduce_bias = True

        try:
            out_bet = bet.run()
        except Exception as e:
            logging.error(e)

        self.output_file = out_bet.outputs.out_file
        # os.system(f"mv {self.output_file} {self.name+'.nii'}")

        self.binary_mask = out_bet.outputs.mask_file
        os.system(f"mv {self.binary_mask} {self.name+'_bm.nii'}")
        self.binary_mask = self.name + '_bm.nii'
        img = load(self.output_file)
        return img.get_fdata(), img.affine, img.header

    def get_mask(self):
        img = load(self.binary_mask)
        return img.get_fdata()