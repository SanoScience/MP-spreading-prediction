''' Generate tractogram using FA threshold or ACT stopping criterion. 
Compute and visualize connectivity matrix. '''

import os
import logging

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import (LocalTracking,
                                          ParticleFilteringTracking)
from dipy.tracking.stopping_criterion import (ThresholdStoppingCriterion, 
                                              ActStoppingCriterion)
from dipy.tracking.streamline import Streamlines, length
from dipy.segment.mask import median_otsu
import nibabel
import yaml
import numpy as np

from generate_connectivity_matrix import ConnectivityMatrix
from utils import parallelize

def get_paths(config, subject_name):
    ''' Generate paths based on configuration file and selected subject. '''

    subject_dir = os.path.join(config['paths']['dataset_dir'], 
                               subject_name)
    img_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                            subject_name+'_ses-1_acq-AP_dwi.nii.gz')
    bval_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                             subject_name+'_ses-1_acq-AP_dwi.bval')
    bvec_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                             subject_name+'_ses-1_acq-AP_dwi.bvec')
    output_dir = os.path.join(config['paths']['output_dir'], 
                              subject_name)
      
    # CerebroSpinal Fluid (CSF) is _pve_0
    csf_path = os.path.join(subject_dir, 'ses-1', 't1', 
                            subject_name+'_ses-1_acq-AP_t1_pve_0.nii.gz')
    
    # Grey Matter is _pve_1
    gm_path = os.path.join(subject_dir, 'ses-1', 't1', 
                           subject_name+'_ses-1_acq-AP_t1_pve_1.nii.gz')
    
    # White Matter is _pve_2
    wm_path = os.path.join(subject_dir, 'ses-1', 't1', 
                           subject_name+'_ses-1_acq-AP_t1_pve_2.nii.gz')

    # AAL atlas path
    atlas_path = config['paths']['atlas_path']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return (img_path, bval_path, bvec_path, output_dir, 
            csf_path, gm_path, wm_path, atlas_path)
    

def get_gradient_table(bval_path, bvec_path):
    ''' Read .bval and .bec files to build the gradient table'''
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

def fa_method(config, data, white_matter, gradient_table, affine, seeds, shm_coeff):
    # apply threshold stopping criterion
    csa_model = CsaOdfModel(gradient_table, sh_order=config['sh_order']) 
    gfa = csa_model.fit(data, mask=white_matter).gfa
    stopping_criterion = ThresholdStoppingCriterion(gfa, config['stop_thres'])
    
    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
        shm_coeff, max_angle=30., sphere=default_sphere)
    streamline_generator = LocalTracking(detmax_dg, stopping_criterion, 
                                        affine=affine,
                                        seeds=seeds,
                                        max_cross=config['max_cross'],
                                        step_size=config['step_size'],
                                        return_all=False)
    return streamline_generator

def act_method(data_wm, data_gm, data_csf, affine, seeds, shm_coeff):
    # anatomical constraints
    dg = ProbabilisticDirectionGetter.from_shcoeff(shm_coeff,
                                            max_angle=20.,
                                            sphere=default_sphere)
    
    act_criterion = ActStoppingCriterion.from_pve(data_wm, data_gm, data_csf)

    # Particle Filtering Tractography
    streamline_generator = ParticleFilteringTracking(dg,
                                                    act_criterion,
                                                    seeds,
                                                    affine,
                                                    max_cross=1,
                                                    step_size=0.2,
                                                    maxlen=1000,
                                                    pft_back_tracking_dist=2,
                                                    pft_front_tracking_dist=1,
                                                    particle_count=15,
                                                    return_all=False)
    return streamline_generator

def remove_short_connections(streamlines, len_thres):
    ''' Filter streamlines with the length shorter 
    than provided threshold [usually in mm]. '''
    logging.info(f'No. of streamlines BEFORE length filtering: {len(streamlines)}')
    longer_streamlines = [s for s in streamlines
                          if compute_streamline_length(s)>=len_thres]
    logging.info(f'No. of streamlines AFTER length filtering: {len(longer_streamlines)}')
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

def generate_tractogram(config, data, affine, hardi_img, gtab, 
                        data_wm, data_gm, data_csf):
    cfg = config['tractogram_config']
    
    # create binary mask based on the first volume
    mask, binary_mask = median_otsu(data[:, :, :, 0]) 
    seed_mask = binary_mask 
    white_matter  = mask 
    seeds = utils.seeds_from_mask(seed_mask, affine, density=cfg['seed_density'])

    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=cfg['fa_thres'])

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=cfg['sh_order'])  
    csd_fit = csd_model.fit(data, mask=white_matter)

    if cfg['stop_method'] == 'FA':
        streamline_generator = fa_method(cfg, data, white_matter, gtab, 
                                         affine, seeds, csd_fit.shm_coeff)  
    elif cfg['stop_method'] == 'ACT':
        streamline_generator = act_method(data_wm, data_gm, data_csf, 
                                          affine, seeds, csd_fit.shm_coeff)
    else:
        logging.error('Provide valid stopping criterion!')
        exit()

    streamlines = remove_short_connections(Streamlines(streamline_generator, cfg['stream_max_len']))

    # generate and save tractogram 
    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    return sft

def save_tractogram(tractogram, output_dir, image_path):
    file_stem = os.path.basename(image_path).split('.')[0]
    save_trk(tractogram, os.path.join(output_dir, f"tractogram_{file_stem}_ACT.trk"))
    logging.info(f"Current tractogram saved as {output_dir}tractogram_{file_stem}_ACT.trk")

def run(config=None, subject_name=None):
    ''' Run workflow for selected subject. '''
    
    # get paths
    (img_path, bval_path, bvec_path, output_dir, 
     csf_path, gm_path, wm_path, atlas_path) = get_paths(config, subject_name)

    # load data 
    data, affine, hardi_img = load_nifti(img_path, return_img=True) 
    data_wm = load_nifti_data(wm_path)
    data_gm = load_nifti_data(gm_path)
    data_csf = load_nifti_data(csf_path)
    gradient_table = get_gradient_table(bval_path, bvec_path)
    
    logging.info(f"Generating tractogram using: {config['tractogram_config']['stop_method']} method")
    logging.info(f"Processing subject: {subject_name}")
    logging.info(f"No. of volumes: {data.shape[-1]}")

    # generate tractogram
    tractogram = generate_tractogram(config, data, affine, hardi_img, 
                                     gradient_table, data_wm, data_gm, data_csf)
    save_tractogram(tractogram, output_dir, img_path)
    
    # generate connectivity matrix
    atlas, labels  = load_atlas(atlas_path)
    cm = ConnectivityMatrix(tractogram, labels, output_dir, 
                            config['tractogram_config']['take_log'])
    cm.process()
      
def main():
    logging.basicConfig(level=logging.INFO)

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
                    
    subject_name = config['paths']['subject']
    
    if subject_name is not None:
        run(config, subject_name)
    else:
        # run workflow on all subjects 
        subject_name_list = os.listdir(config['paths']['dataset_dir'])
        parallelize(subject_name_list, run, config)

if __name__ == '__main__':
    main()