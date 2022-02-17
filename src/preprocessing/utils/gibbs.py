from nibabel import save, Nifti1Image
from dipy.denoise.gibbs import gibbs_removal
import logging

class Gibbs:
    data = None
    header = None 
    affine = None
    name = None
    
    def __init__(self, data, affine, header, name):
        self.data = data
        self.header = header
        self.affine = affine
        self.name = name

    def run(self):
        self.data = gibbs_removal(self.data)
        return self.data, self.affine, self.header