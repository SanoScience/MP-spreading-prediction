''' Optimize Tractogram '''

import os
import logging
from dipy.io.streamline import save_trk, load_trk
import dipy.tracking.life as life
from dipy.tracking.life import unique_rows
from dipy.io.streamline import load_tractogram, save_tractogram
import yaml
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import length


def get_paths(config):
    # Directory containing tractogram to optimize
    output_path = os.path.join(config['paths']['output_dir'], 
                                config['paths']['subject'])
    
    subject_path = os.path.join(config['paths']['dataset_dir'], 
                               config['paths']['subject'])
    tractogram_path = os.path.join(output_path, 
                                   f"tractogram_{config['paths']['subject']}_ses-1_acq-AP_dwi_ACT.trk")
    img_path = os.path.join(subject_path, 'ses-1', 'dwi', 
                            config['paths']['subject']+'_ses-1_acq-AP_dwi.nii.gz')

    bval_path = os.path.join(subject_path, 'ses-1', 'dwi', 
                                config['paths']['subject']+'_ses-1_acq-AP_dwi.bval')
    bvec_path = os.path.join(subject_path, 'ses-1', 'dwi', 
                                config['paths']['subject']+'_ses-1_acq-AP_dwi.bvec')

    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    return output_path, subject_path, tractogram_path, img_path, bval_path, bvec_path

# TODO: make 'generate_tractogram.py' to save the gradient table to easy access from this script
def get_gradient_table(bval_path, bvec_path):
    ''' Read .bval and .bec files to build the gradient table'''
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

def remove_short_connections(streamlines, len_thres=20):
    ''' Filter streamlines with the length shorter 
    than provided threshold [usually in mm]. '''
    logging.info(f'No. of streamlines before length filtering: {len(streamlines)}')
    longer_streamlines = [s for s in streamlines
                          if compute_streamline_length(s)>len_thres]
    logging.info(f'No. of streamlines after length filtering: {len(longer_streamlines)}')
    logging.info(f'Percentage of remaining streamlines: {round(len(longer_streamlines)/len(streamlines), 4)*100}')
    return longer_streamlines

def compute_streamline_length(streamline, is_dipy=True):
    if is_dipy:
        # use built-in DIPY function to calculate the length of streamline
        return length(streamline)
    else:
        # use classical method with calculating vector norm
        streamline = np.array(streamline)
        s_length = np.sum([np.linalg.norm(streamline[i+1]-streamline[i]) 
                    for i in range(0, len(streamline)-1)])
    return s_length

def save_tractogram(tractogram, output_dir, image_path):
    file_stem = os.path.basename(image_path).split('.')[0]
    #tractogram.streamlines = list(np.array(tractogram.streamlines, dtype=object)[np.where(tractogram.remove_invalid_streamlines())[1]])
    tractogram.to_rasmm()
    save_trk(tractogram, os.path.join(output_dir, f"opt_tractogram_{file_stem}_ACT.trk"), bbox_valid_check=False)
    logging.info(f"Current tractogram saved as {output_dir}opt_tractogram_{file_stem}_ACT.trk")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logging.info("Config file loaded")
    
    # get paths
    output_path, subject_path, tractogram_path, img_path, bval_path, bvec_path  = get_paths(config)
    data, affine, hardi_img = load_nifti(img_path, return_img=True)
    logging.info("Paths read")
    
    file_stem = os.path.basename(tractogram_path).split('.')[0]
    # TRK files contain their own header (when wrote properly), so they technically do not need a reference. (See how below) cc_trk = load_tractogram(bundles_filename[0], ‘same’)
    tractogram = load_tractogram(tractogram_path, 'same')
    logging.info(f"Tractogram loaded: {tractogram}")
    
    #gtab = get_gradient_table(bval_path, bvec_path)
    logging.info("Gradient table computed") 
    
    #tractogram.to_vox()
    logging.info("Tractogram converted into Voxels Space")
        
    # Optimization phase (Linear Fascicle Evaluation, LiFE https://dipy.org/documentation/1.1.1./examples_built/linear_fascicle_evaluation/)    
    #fiber_model = life.FiberModel(gtab)
    #logging.info("Model created")
    #Length will return the length in the units of the coordinate system that streamlines are currently. So, if the streamlines are in world coordinates then the lengths will be in millimeters (mm). If the streamlines are for example in native image coordinates of voxel size 2mm isotropic then you will need to multiply the lengths by 2 if you want them to correspond to mm
    candidate_sl = remove_short_connections(tractogram.streamlines)
    #candidate_sl = unique_rows(np.asarray(tractogram.streamlines))
    logging.info(f"Putting {len(candidate_sl)} fibers into the model")
    
    # freeing memory
    #del tractogram
    #del gtab
    
    # LiFE algorithm evaluates how well the entire connectome fits the white-matter diffusion data. 
    # Streamlines are used to fit a linear model able to evaluate tractography results and assign a weight (Beta) to each streamline, representing its expected contribution (redundant streamlines have a weight of 0)  
    #fiber_fit = fiber_model.fit(data, candidate_sl, affine=np.eye(4))
    logging.info("Model fitted")
    
    #del data
    
    # redundant fibers (beta = 0) are removed
    #opt_streamlines = list(np.array(candidate_sl, dtype=object)[np.where(fiber_fit.beta > 0)[0]])
    #opt_streamlines = candidate_sl
    #logging.info(f"{len(opt_streamlines)} Optimal streamlines computed")
    
    sft = StatefulTractogram(candidate_sl, hardi_img, Space.RASMM)
    save_tractogram(sft, output_path, img_path)
    

    # Evaluating Root Mean Square Error
    #model_predict = fiber_fit.predict()
    #model_error = model_predict - fiber_fit.data
    #model_rmse = np.sqrt(np.mean(model_error[:, 10:] ** 2, -1))
    #logging.info(f"The Root Mean Square Error between Optimized streamlines and the original ones is: {model_rmse}")