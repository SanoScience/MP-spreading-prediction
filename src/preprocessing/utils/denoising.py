from json import load
from nibabel import save, Nifti1Image
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import logging

class Denoising_LPCA:
    data = None
    header = None 
    affine = None
    binary_mask = None
    bvals = None
    bvecs = None
    gtab = None
    name = None 
    
    def __init__(self, data, affine, header, name, binary_mask, bval_file = None, bvec_file = None):
        self.data = data
        self.header = header
        self.affine = affine
        self.binary_mask = binary_mask
        self.name = name 
        if bvec_file and bval_file: 
            self.bvals, self.bvecs = read_bvals_bvecs(bval_file, bvec_file)

    def run(self, gtab=None):
        if not gtab:
            gtab = gradient_table(self.bvals, self.bvecs)
        sigma = pca_noise_estimate(self.data, gtab, correct_bias = True, smooth=2, patch_radius = 2)
        self.data = localpca(self.data, sigma, self.binary_mask, tau_factor=2.3, patch_radius = 2) 
        # tau_factor = None implies the automatic computation of threshold for PCA eigenvalues
        # patch_radius too big will result in a smaller but less refined image
        return self.data, self.affine, self.header