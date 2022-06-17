from nibabel import load, save, Nifti1Image
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs

from nipype.interfaces import fsl
import logging
import os

class Registration():
    def __init__(self, name_nii, ref, output_dir, name, img_type):
        self.name_nii = name_nii
        self.ref = ref
        self.name = name
        self.img_type = img_type
        self.output_dir = output_dir

    def run(self):
        
        if self.img_type == 'dwi': # multiple volumes images are truncated to the first volume to do the registration (then the affine matrix is applied to the whole image)
            input_reg = 'first_slice.nii.gz'
            img = load(self.name_nii)
            data, affine, header = img.get_fdata()[:,:,:,0], img.affine, img.header
            save(Nifti1Image(data, affine, header), input_reg)
        else:
            input_reg = self.name_nii
            
        matrix_name = self.output_dir + self.name + '_reg_matrix.mat'
        
        fl = fsl.FLIRT()
        fl.inputs.in_file = input_reg
        fl.inputs.reference = self.ref
        fl.inputs.out_file = self.output_dir + self.name + '_reg.nii.gz'
        fl.inputs.output_type = 'NIFTI_GZ'
        # NOTE: DO NOT CHANGE BINS
        #fl.inputs.bins = 1500
        fl.inputs.dof = 12
        fl.inputs.cost_func = 'mutualinfo'
        # For 3D to 3D mode the DOF can be set to 12 (affine), 9 (traditional), 7 (global rescale) or 6 (rigid body)
        fl.inputs.out_matrix_file = matrix_name
        
        # Produce the registration matrix on the first volume of the image...
        try:
            out_fl = fl.run()
        except Exception as e:
            logging.error(e)
            logging.error(f"{self.name_nii} at Registration.run()")
            print(e)
            print(f"{self.name_nii} at Registration.run()")
        
        # After registration matrix is computed, pass the whole image to which apply the registration matrix
        if self.img_type == 'dwi':
            fl.inputs.in_file = self.name_nii
            os.system(f"rm {input_reg}")
            
        # Anatomical and PET images are done with the first invocation, DWIs need two 
        if self.img_type == 'dwi':
            # ... and use the matrix to register all the volumes
            fl.inputs.apply_xfm = True
            fl.inputs.in_matrix_file = matrix_name
            
            try:
                out_fl = fl.run()
            except Exception as e:
                logging.error(e)

            # reorient bvectors
            reg_mat = open(matrix_name)
            bval_f = self.name + '.bval'
            bvec_f = self.name + '.bvec'
            bvals, bvecs = read_bvals_bvecs(bval_f, bvec_f)
            gtab = gradient_table(bvals, bvecs)
            vols = gtab.bvals.shape[0]
            v_trans = np.loadtxt(reg_mat)
            trans = []
            for v in range(vols):
                if not gtab.b0s_mask[v]:
                    trans.append(v_trans) 
            gtab_corr = reorient_bvecs(gtab, trans)
            np.savetxt(bvec_f, gtab_corr.bvecs.T)
            
        registered = out_fl.outputs.out_file
        img = load(registered)
        
        return img.get_fdata(), img.affine, img.header