''' Generate tractogram using FA threshold or ACT stopping criterion. '''

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
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
from nibabel.affines import voxel_sizes
import yaml
import numpy as np

from utils import task_completion_info

def get_paths(config):
    ''' Generate paths based on configuration file. '''
    subject_dir = os.path.join(config['paths']['dataset_dir'], 
                               config['paths']['subject'])
    img_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                            config['paths']['subject']+'_ses-1_acq-AP_dwi.nii.gz')
    bval_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                             config['paths']['subject']+'_ses-1_acq-AP_dwi.bval')
    bvec_path = os.path.join(subject_dir, 'ses-1', 'dwi', 
                             config['paths']['subject']+'_ses-1_acq-AP_dwi.bvec')
    output_dir = os.path.join(config['paths']['output_dir'], 
                              config['paths']['subject'])
      
    # CerebroSpinal Fluid (CSF) is _pve_0
    csf_path = os.path.join(output_dir,
                           config['paths']['subject']+'_pve_0.nii.gz')
    
    # Grey Matter is _pve_1
    gm_path = os.path.join(output_dir,
                           config['paths']['subject']+'_pve_1.nii.gz')
    
    # White Matter is _pve_2
    wm_path = os.path.join(output_dir,
                           config['paths']['subject']+'_pve_2.nii.gz')
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return img_path, bval_path, bvec_path, output_dir, csf_path, gm_path, wm_path
    

def get_gradient_table(bval_path, bvec_path):
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

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

    streamlines = Streamlines(streamline_generator)

    # generate and save tractogram 
    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    return sft

def save_tractogram(tractogram, output_dir, image_path):
    file_stem = os.path.basename(image_path).split('.')[0]
    save_trk(tractogram, os.path.join(output_dir, f"tractogram_{file_stem}_ACT.trk"))

def main():
    logging.basicConfig(level=logging.INFO)

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get paths
    img_path, bval_path, bvec_path, output_dir, csf_path, gm_path, wm_path = get_paths(config)

    # load data 
    data, affine, hardi_img = load_nifti(img_path, return_img=True) 
    data_wm = load_nifti_data(wm_path)
    data_gm = load_nifti_data(gm_path)
    data_csf = load_nifti_data(csf_path)
    gradient_table = get_gradient_table(bval_path, bvec_path)
    
    print(data_wm.shape, data_gm.shape, data_csf.shape)

    logging.info(f"Generating tractogram using: {config['tractogram_config']['stop_method']} method")
    logging.info(f"Processing subject: {config['paths']['subject']}")
    logging.info(f"No. of volumes: {data.shape[-1]}")

    tractogram = generate_tractogram(config, data, affine, hardi_img, 
                                     gradient_table, data_wm, data_gm, data_csf)
    save_tractogram(tractogram, output_dir, img_path)

    task_completion_info()

if __name__ == '__main__':
    main()