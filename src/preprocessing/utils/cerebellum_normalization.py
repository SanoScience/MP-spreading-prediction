from nipype.interfaces.fsl.maths import Threshold, ApplyMask, MathsCommand
from nipype.interfaces.fsl import ImageStats
from nibabel import load
import os
import logging

class CerebellumNormalization:
    name_nii = None
    atlas = None
    final_out_name = None
    name = None
    
    def __init__(self, name_nii, atlas_path, name):
        self.name_nii = name_nii
        self.atlas = atlas_path
        self.name = name

    def run(self):
        
        # Center the values to an average of 1
        stats = ImageStats()
        stats.inputs.in_file = self.name_nii
        stats.inputs.op_string = '-M'
        out_stats = stats.run()
        total_mean = out_stats.outputs.out_stat
        math = MathsCommand()
        math.inputs.in_file = self.name_nii
        math.inputs.args = f'-sub {total_mean - 1} -thr 0'
        math.inputs.out_file =  self.name + '.nii'
        math.inputs.output_type = 'NIFTI'
        out_math = math.run()
        flattened_path = out_math.outputs.out_file
        
        # --------------------


        # Create a mask for the cerebellum, using the atlas
        thr = Threshold()
        thr.inputs.in_file = self.atlas
        thr.inputs.thresh = 94.9
        # threshold on values below 95 (non-cerebellum voxels are cut off)
        thr.inputs.direction = 'below'
        thr.inputs.nan2zeros = True
        thr.inputs.out_file = self.name + '_cb-mask.nii'
        thr.inputs.output_type = 'NIFTI'
        
        try:
            out_thr = thr.run()
        except Exception as e:
            logging.error(e)
            
        cerebellum_mask_path = out_thr.outputs.out_file
        
        # again, now removing regions above 112
        thr.inputs.in_file = cerebellum_mask_path
        thr.inputs.thresh = 112.1
        thr.inputs.direction = 'above'
        thr.inputs.nan2zeros = True
        thr.inputs.out_file = self.name + '_cb-mask.nii'
        thr.inputs.output_type = 'NIFTI'
        
        try:
            out_thr = thr.run()
        except Exception as e:
            logging.error(e)
            
        cerebellum_mask_path = out_thr.outputs.out_file
        
        # --------------------

        # Apply the mask for the cerebellum, created through the atlas, to the flattened image
        am = ApplyMask()
        am.inputs.in_file = flattened_path
        am.inputs.mask_file = cerebellum_mask_path
        am.inputs.out_file = self.name + '_cb-only.nii'
        am.inputs.output_type = 'NIFTI'
        out_am = am.run()
        cerebellum_image_path = out_am.outputs.out_file
        
        # --------------------

        # Compute the mean of cerebellum voxels and substract to the flattened image
        stats = ImageStats()
        stats.inputs.in_file = cerebellum_image_path
        stats.inputs.op_string = '-M'
        out_stats = stats.run()
        cerebellum_mean = out_stats.outputs.out_stat
        
        # --------------------
    
        # subtract cerebellum_mean to all voxels in the flattened image and truncate negative values to 0
        math = MathsCommand()
        math.inputs.in_file = flattened_path
        math.inputs.args = f'-sub {cerebellum_mean} -thr 0'
        math.inputs.out_file =  self.name + '.nii'
        math.inputs.output_type = 'NIFTI'
        out_math = math.run()
        flattened_normalized_path = out_math.outputs.out_file

        '''
        echo "subtracting mean of cerebellum to the rest"
        fslmaths $output -sub `fslstats $cerebellum_image -M` $output

        echo "removing negative values"
        fslmaths $output -thr 0 $output
        '''
        img = load(flattened_normalized_path)

        return img.get_fdata(), img.affine, img.header