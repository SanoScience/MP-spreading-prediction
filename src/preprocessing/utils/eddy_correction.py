import shutil
from nibabel import load
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import json
import os
import gzip
import logging

class EddyMotionCorrection:
    out_image = ''
    out_bvecs = ''

    def __init__(self, name, name_nii, name_bval, name_bvec, name_json, mask_path):
        # Binary mask for skull is mandatory, so input file has been surely submitted to 'brain_extraction' module

        self.name = name 
        self.name_nii = name_nii
        
        self.index_path = self.name + "_index.txt"
        self.acqp_path = self.name + "_acqparams.txt"

        self.bval_path = name_bval
        self.bvec_path = name_bvec
        self.mask_path = mask_path
        self.json_path = name_json
        self.base = 'eddy_corrected'

        # The acqparams.txt is manually writen if the fields are not present in the json file
        json_data = json.load(open(self.json_path))
        try:
            image_orientation = json_data["ImageOrientationPatientDICOM"]
            amPE = json_data["AcquisitionMatrixPE"]
        except: 
            image_orientation = np.eye(2,3).flatten()
            amPE =  "KeyNotPresent"
        es_codename = "EffectiveEchoSpacing"
        if es_codename not in json_data.keys():
            es_codename = "Estimated"+es_codename
            if es_codename not in json_data.keys():
                es_codename = "EchoTime"
        ees = json_data[es_codename]
        try:
            amPE = json_data["AcquisitionMatrixPE"]
            freq = round(ees * (amPE - 1), 4)
        except: 
            freq = 0.043
        
        #TODO: read PhaseEncodingDirection and put the right row in the acq_f (i.e. 'j' is [0 1 0])
            
        acq_f = open(self.acqp_path, "w")
        acq_f.write(f"{image_orientation[0]} {image_orientation[1]} {image_orientation[2]} {freq}\n")
        acq_f.write(f"{image_orientation[3]} {image_orientation[4]} {image_orientation[5]} {freq}\n")
        acq_f.close()
        
        index = open(self.index_path, "w")
        bvals = open(self.bval_path, "r")
        volumes = len(bvals.readline().split(' '))
        for v in range(volumes):
            index.write('1 ' if v%2==0 else '2 ')
        index.close()

    def run(self):        
        os.system(f"eddy --imain={self.name_nii} --mask={self.mask_path} --acqp={self.acqp_path} --index={self.index_path} --bvecs={self.bvec_path} --bvals={self.bval_path} --out={self.base} --repol --interp=trilinear --niter=3 --nvoxhp=500")
        '''
        The --out parameter specifies the basename for all output files of eddy. It is used as the name for all eddy output files, but with different extensions. If we assume that user specified --out=my_eddy_output, the files that are always written are

            my_eddy_output.nii.gz
            This is the main output and consists of the input data after correction for eddy currents and subject movement, and for susceptibility if --topup or --field was specified, and for signal dropout if --repol was set. Chances are this is the only output file you will be interested in (in the context of eddy).

            my_eddy_output.eddy_rotated_bvecs
            When a subject moves such that it constitutes a rotation around some axis and this is subsequently reoriented, it will create an inconsistency in the relationship between the data and the "bvecs" (directions of diffusion weighting). This can be remedied by using the my_eddy_output.rotated_bvecs file for subsequent analysis. For the rotation to work correctly the bvecs need to be "correct" for FSL before being fed into eddy. The easiest way to check that this is the case for your data is to run FDT and display the _V1 files in fslview or FSLeyes to make sure that the eigenvectors line up across voxels.
        '''
        self.out_image = 'eddy_corrected.nii.gz'
        self.out_bvecs = 'eddy_corrected.eddy_rotated_bvecs'

        # extracting .nii.gz
        with gzip.open(self.out_image, 'rb') as f_in:
            with open(self.name+'.nii', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)    
        self.out_image = self.name+'.nii'

        os.system(f"mv {self.out_bvecs} {self.name+'.bvec'}")
        self.out_bvecs = self.name+'.bvec'

        os.system(f"cp {self.bval_path} .")

        img = load(self.out_image)
        
        #logging.info("Removing intermediate eddy files")
        
        return img.get_fdata(), img.affine, img.header
    
    def get_BMatrix(self):
        bvals, bvecs = read_bvals_bvecs(self.name + '.bval', self.out_bvecs)
        return gradient_table(bvals, bvecs) 
    
    def get_bvec_bval(self):            
        return self.out_bvecs, self.name + '.bval'