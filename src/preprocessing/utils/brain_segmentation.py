import nibabel as nib
import logging
from nipype.interfaces.fsl import FAST
from pathlib import Path
import os

class BrainSegmentation:
    def __init__(self, name_nii, name):
        self.name_nii = name_nii
        self.name = name

    def run(self, nclass=3, img_type='t1', b1_corr=False):
        if img_type == 't1':
            type_seg = 1
        else:
            type_seg = 2

        fst = FAST()
        fst.inputs.in_files = self.name_nii
        fst.inputs.img_type = type_seg
        fst.inputs.no_bias = b1_corr
        fst.inputs.no_pve = False
        fst.inputs.number_classes = nclass
        fst.inputs.output_type = 'NIFTI'
        fst.inputs.out_basename = self.name

        try:
            out_fst = fst.run()
        except Exception as e:
            logging.error(e)
        
        self.seg = out_fst.outputs.tissue_class_map
        self.pves = out_fst.outputs.partial_volume_files
        
        name_p = self.name+'_pve-{}.nii'
        Path(self.name+'_pve_0.nii').rename(name_p.format('0'))
        Path(self.name+'_pve_1.nii').rename(name_p.format('1'))
        Path(self.name+'_pve_2.nii').rename(name_p.format('2'))
                
        img = nib.load(self.seg)
        return img.get_fdata(), img.affine, img.header