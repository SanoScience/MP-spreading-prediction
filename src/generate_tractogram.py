import os

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
import yaml

def get_paths(config):
    ''' Generate paths based on configuration file. '''
    subject_dir = os.path.join(config['paths']['dataset_dir'], config['paths']['subject'])
    img_path = os.path.join(subject_dir, 'ses-1', 'dwi', config['paths']['subject']+'_ses-1_acq-AP_dwi.nii.gz')
    bval_path = os.path.join(subject_dir, 'ses-1', 'dwi', config['paths']['subject']+'_ses-1_acq-AP_dwi.bval')
    bvec_path = os.path.join(subject_dir, 'ses-1', 'dwi', config['paths']['subject']+'_ses-1_acq-AP_dwi.bvec')
    output_dir = os.path.join(config['paths']['output_dir'], config['paths']['subject'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return img_path, bval_path, bvec_path, output_dir

def load_nifti_data(img_path):
    data, affine, hardi_img = load_nifti(img_path, return_img=True) 
    return data, affine, hardi_img

def get_gradient_table(bval_path, bvec_path):
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

def generate_tractogram(config, data, affine, hardi_img, gtab):
    cfg = config['tractogram_config']

    # create binary mask based on the first volume
    mask, binary_mask = median_otsu(data[:, :, :, 0]) 
    seed_mask = binary_mask 
    white_matter  = mask 
    seeds = utils.seeds_from_mask(seed_mask, affine, density=cfg['seed_density'])

    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=cfg['fa_thres'])

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=cfg['sh_order'])  
    csd_fit = csd_model.fit(data, mask=white_matter)

    csa_model = CsaOdfModel(gtab, sh_order=cfg['sh_order']) 
    gfa = csa_model.fit(data, mask=white_matter).gfa
    stopping_criterion = ThresholdStoppingCriterion(gfa, cfg['stop_thres'])

    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
        csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
    streamline_generator = LocalTracking(detmax_dg, stopping_criterion, 
                                        affine=affine,
                                        seeds=seeds,
                                        max_cross=cfg['max_cross'],
                                        step_size=cfg['step_size'],
                                        return_all=False)
    streamlines = Streamlines(streamline_generator)

    # generate and save tractogram 
    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    return sft

def save_tractogram(tractogram, output_dir, image_path):
    file_stem = os.path.basename(image_path).split('.')[0]
    save_trk(tractogram, os.path.join(output_dir, f"tractogram_{file_stem}.trk"))

def task_completion_info(sound_duration=1, sound_freq=440):
    print('Tractograms generated. Process completed.')
    os.system(f'play -nq -t alsa synth {sound_duration} sine {sound_freq}')

def main():
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_path, bval_path, bvec_path, output_dir = get_paths(config)

    data, affine, hardi_img = load_nifti_data(img_path)
    gradient_table = get_gradient_table(bval_path, bvec_path)

    print(f"[INFO] Processing subject: {config['paths']['subject']}")
    print(f"[INFO] No. of volumes: {data.shape[-1]}")

    tractogram = generate_tractogram(config, data, affine, hardi_img, gradient_table)
    save_tractogram(tractogram, output_dir, img_path)

    task_completion_info()

if __name__ == '__main__':
    main()