''' Generate tractogram using FA threshold or ACT stopping criterion. 
Compute and visualize connectivity matrix. '''

from datetime import datetime
import os
import logging
import sys

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, load_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking import utils
from dipy.tracking.local_tracking import ParticleFilteringTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion, ActStoppingCriterion
from dipy.tracking.streamline import Streamlines, length
import nibabel
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils import parallelize
from glob import glob

class ConnectivityMatrix():
    def __init__(self, tractogram, atlas_labels, output_dir):
        self.streamlines = tractogram.streamlines
        self.affine = tractogram.affine # transformation to align streamlines to atlas 
        self.labels = atlas_labels  
        self.output_dir = output_dir
                       
    def __create(self):
        ''' Get the no. of connections between each pair of brain regions. '''
        M, _ = utils.connectivity_matrix(self.streamlines, 
                                        affine=self.affine, 
                                        label_volume=self.labels, 
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
        # remove background
        M = M[1:, 1:]

        # remove connections to own regions (inplace)
        np.fill_diagonal(M, 0) 
        self.matrix = M

        
    def __sort(self):
        # make all left areas first 
        odd_odd = self.matrix[::2, ::2]
        odd_even = self.matrix[::2, 1::2]
        first = np.vstack((odd_odd, odd_even))
        even_odd = self.matrix[1::2, ::2]
        even_even= self.matrix[1::2, 1::2]
        second = np.vstack((even_odd, even_even))
        self.matrix = np.hstack((first,second))
        
    def __save(self, name='connect_matrix.csv'):
        np.savetxt(os.path.join(self.output_dir, name), 
                   self.matrix, delimiter=',')

    def __plot(self, savefig=True):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.matrix, interpolation='nearest')
        plt.colorbar()
        plt.title(f'Connectivity matrix')
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(self.output_dir, 'connect_matrix.png'))
        
    def __get_info(self):
        logging.info(f'Shape of connectivity matrix: {self.matrix.shape}. \
        Sum of values: {np.sum(self.matrix)} (after removing background and connections to own regions)')
        
    def process(self, reshuffle = True):
        self.__create()   
        self.__get_info()

        self.__save('connect_matrix_original.csv') 

        self.matrix = self.matrix/np.max(self.matrix)
        self.__save('connect_matrix_norm.csv')

        if reshuffle:
            self.__sort() # sort rois to make rois 'left-to-right' oriented in the matrix 
            self.__save('connect_matrix_sort.csv') # 'Sort' is the matrix meant to be used in BrainNetViewer and manual analysis
        
        self.__plot()     

    def __str__(self) -> str:
        return self.output_dir

def get_paths(stem_dwi, stem_anat, config, general_dir):
    ''' Generate paths based on configuration file and selected subject. '''
 
    img_path = stem_dwi + '.nii.gz'
    bval_path = stem_dwi + '.bval'
    bvec_path = stem_dwi + '.bvec'
    # outputs are saved in the same folder of dwi files
    output_dir = stem_dwi.rstrip(stem_dwi.split(os.sep)[-1])
      
    bm_path = stem_dwi + '_mask.nii.gz'      
    # CerebroSpinal Fluid (CSF) is _pve_0
    csf_path = stem_anat + '_pve-0.nii.gz'    
    # Grey Matter is _pve_1
    gm_path = stem_anat + '_pve-1.nii.gz'    
    # White Matter is _pve_2
    wm_path = stem_anat + '_pve-2.nii.gz'

    return (img_path, bval_path, bvec_path, output_dir, bm_path,
            csf_path, gm_path, wm_path)
    

def get_gradient_table(bval_path, bvec_path):
    ''' Read .bval and .bec files to build the gradient table'''
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

"""
NOTE: DEPRECATED
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
"""

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
    #logging.info(f'No. of streamlines BEFORE length filtering: {len(streamlines)}')
    longer_streamlines = [s for s in streamlines
                          if compute_streamline_length(s)>=len_thres]
    #logging.info(f'No. of streamlines AFTER length filtering: {len(longer_streamlines)}')
    logging.info(f'Percentage of remaining streamlines after filtering: {round(len(longer_streamlines)/len(streamlines), 4)*100}')
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

def generate_tractogram(config, data, affine, hardi_img, gtab, data_bm,
                        data_wm, data_gm, data_csf):
    try:
        seeds = utils.seeds_from_mask(data_bm, affine, density=config['tractogram_config']['seed_density'])
    except Exception as e:
        logging.error("Error at 'seeds_from_mask'. Traceback:")
        logging.error(e)
    try:
        response, _ = auto_response_ssst(gtab, data, roi_radii=20, fa_thr=config['tractogram_config']['fa_thres'])
    except Exception as e:
        logging.error("Error at 'auto_response_ssst'. Traceback:")
        logging.error(e)

    try:
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, convergence=500, sh_order=config['tractogram_config']['sh_order'])  
        csd_fit = csd_model.fit(data, mask=data_bm)
    except Exception as e:
        logging.error("Error at 'ConstrainedSphericalDeconvModel'. Traceback:")
        logging.error(e)

    """
    NOTE: DEPRECATED
    if config['tractogram_config']['stop_method'] == 'FA':
        streamline_generator = fa_method(config['tractogram_config'], data, data_bm, gtab, 
                                         affine, seeds, csd_fit.shm_coeff)  
    """
    if config['tractogram_config']['stop_method'] == 'ACT':
        try:
            streamline_generator = act_method(data_wm, data_gm, data_csf, 
                                          affine, seeds, csd_fit.shm_coeff)
        except Exception as e:
            logging.error("Error at 'act_method'. Traceback:")
            logging.error(e)
    else:
        logging.error('Provide valid stopping criterion!')
        exit()

    streamlines = Streamlines(streamline_generator)
    try:
        streamlines = remove_short_connections(streamlines, config['tractogram_config']['stream_min_len'])
    except Exception as e:
        logging.error("Error at 'remove_short_connections'. Traceback:")
        logging.error(e)
    
    sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
    return sft

def save_tractogram(tractogram, output_dir, image_path):
    file_stem = os.path.basename(image_path).split('.')[0]
    save_trk(tractogram, os.path.join(output_dir, f"{file_stem}_sc-act.trk"))
    logging.info(f"Current tractogram saved as {output_dir}{file_stem}_sc-act.trk")
    
def load_tractogram(tractogram):
    return load_trk(tractogram, 'same')

def run(stem_dwi = '', stem_anat = '', tractogram_file = '', config = None, general_dir = ''):
    ''' Run workflow for selected subject. '''
    # AAL atlas path
    atlas_path = general_dir + config['paths']['atlas_path']
    
    if tractogram_file == '':
        # get paths
        img_path, bval_path, bvec_path, output_dir, bm_path, csf_path, gm_path, wm_path = get_paths(stem_dwi, stem_anat, config, general_dir)

        # load data 
        data, affine, hardi_img = load_nifti(img_path, return_img=True) 
        data_bm = load_nifti_data(bm_path)
        data_wm = load_nifti_data(wm_path)
        data_gm = load_nifti_data(gm_path)
        data_csf = load_nifti_data(csf_path)
        gradient_table = get_gradient_table(bval_path, bvec_path)

        try:
            tractogram = generate_tractogram(config, data, affine, hardi_img, 
                                            gradient_table, data_bm, data_wm, data_gm, data_csf)
            save_tractogram(tractogram, output_dir, img_path)
        except Exception as e:
            logging.error(f"Exception during tractogram generation for file {stem_dwi}")
            logging.error(e)
            f = open('log.txt', 'a')
            f.write(stem_dwi + '\n')
            f.close()
            return
    else:
        output_dir = tractogram_file.rstrip(tractogram_file.split(os.sep)[-1])
        try:
            tractogram = load_tractogram(tractogram_file)
        except Exception as e:
            logging.error(f"Error at 'load_tractogram' for file {tractogram_file}. Traceback:")
            logging.error(e)
            return
    
    # generate connectivity matrix
    try:
        atlas, labels  = load_atlas(atlas_path)
    except Exception as e:
        logging.error("Error at 'load_atlas'. Traceback:")
        logging.error(e)
        return

    try:
        cm = ConnectivityMatrix(tractogram, labels, output_dir)
        cm.process()
        logging.info(f"CM {cm} created")
    except Exception as e:
        logging.error(f"Error at CM creation in {output_dir}. Traceback:")
        logging.error(e)
    return
    

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

if __name__ == '__main__':

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv)> 1:
        dwi_files = open(sys.argv[1]).readlines()
    else:
        os.chdir(os.getcwd() + '/../../..')
        general_dir = os.getcwd() + os.sep
        logging.info(general_dir)
        subject = config['paths']['subject'] if config['paths']['subject'] != 'all' else 'sub-*'
        keep_tract = config['tractogram_config']['keep_tract']
        # NOTE: sub*_dwi.nii.gz won't catch harmonized or other harmonization files, they will be checked later to avoid including intermediate harmonization files
        # NOTE: narrowing tractography on baseline images!
        dwi_dir = general_dir + config['paths']['dataset_dir'] + subject  + os.sep + 'ses-baseline' + os.sep + 'dwi' + os.sep + 'sub*_dwi.nii.gz'
        logging.info(dwi_dir)

        dwi_files = glob(dwi_dir)

    harm_counter = 0
    tract_files = []
    for dwi in dwi_files:
        if keep_tract and os.path.isfile(dwi.replace('.nii.gz', '_sc-act.trk')): 
            logging.info(f"{dwi} has already a tractography, doing only CM")
            dwi_files.remove(dwi)
            tract_files.append(dwi.replace('.nii.gz', '_sc-act.trk'))
        elif 'harmonized' not in dwi:
            k = dwi.rfind('sub')
            harm = dwi[:k] + 'harmonized_' + dwi[k:]
            if os.path.isfile(harm):
                logging.info(f"Harmonized version of {dwi} found, using it instead of the standard one")
                dwi_files.append(harm)
                dwi_files.remove(dwi)
                harm_counter += 1

    logging.info(f'{len(dwi_files)} DWI files to process ({harm_counter} named \'harmonized\')')
    logging.info(dwi_files)
    logging.info(f"{len(tract_files)} tractograms found to do only CM")
    parallelize(dwi_files, tract_files, config['tractogram_config']['cores'], run, config, general_dir)