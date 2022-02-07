''' Generate tractogram using FA threshold or ACT stopping criterion. 
Compute and visualize connectivity matrix. '''

import os
import logging

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import load_tractogram
from dipy.tracking import utils
import nibabel
import yaml
import numpy as np
import matplotlib.pyplot as plt


from generate_connectivity_matrix import ConnectivityMatrix
from utils import parallelize_CM
from glob import glob

class ConnectivityMatrix():
    def __init__(self, tractogram, atlas_labels, output_dir, take_log):
        # NOTE: this is a lazy solution that works assuming you are calling the script inside its folder (src/tractography)
        self.streamlines = tractogram.streamlines
        self.affine = tractogram.affine # transformation to align streamlines to atlas 
        self.labels = atlas_labels  
        self.output_dir = output_dir
        self.take_log = take_log
                       
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
        if self.take_log: M = np.log1p(M)

        self.matrix = M
        
    def __revert(self):
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
        plt.title(f'Connectivity matrix (log values: {self.take_log})')
        plt.tight_layout()
        if savefig: plt.savefig(os.path.join(self.output_dir, 'connect_matrix.png'))
        
    def __get_info(self):
        logging.info(f'Shape of connectivity matrix: {self.matrix.shape}. \
        Sum of values: {np.sum(self.matrix)} (after removing background and connections to own regions)')
        
    def process(self, reshuffle = True):
        self.__create()   
        self.__get_info()
        self.__save('connect_matrix_rough.csv') # 'Rough' means 'as it is', without reverting rois
        if reshuffle:
            self.__revert() # reverts rois to make rois 'left-to-right' oriented in the matrix 
            self.__save('connect_matrix_reverted.csv') # 'Reverted' is the matrix meant to be used in BrainNetViewer and manual analysis
        self.__plot() 

def get_gradient_table(bval_path, bvec_path):
    ''' Read .bval and .bec files to build the gradient table'''
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gradient_tab = gradient_table(bvals, bvecs)
    return gradient_tab

def load_atlas(atlas_path):
    
    atlas = nibabel.load(atlas_path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   


def run(trk_path, config=None, general_dir = '', output_dir = ''):
    ''' Run workflow for selected subject. '''

    # load data 
    atlas_path = general_dir + config['paths']['atlas_path']
    trk = load_tractogram(trk_path, 'same')
    
    # generate connectivity matrix
    atlas, labels  = load_atlas(atlas_path)
    cm = ConnectivityMatrix(trk, labels, output_dir, 
                            config['tractogram_config']['take_log'])
    cm.process()
      
def main():
    #logging.basicConfig(level=logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    general_dir = os.getcwd()
    general_dir = general_dir.removesuffix(os.sep + 'tractography').removesuffix(os.sep + 'src') + os.sep
    subject = config['paths']['subject'] if config['paths']['subject'] != 'all' else 'sub-*'
    trk_dir = general_dir + config['paths']['dataset_dir'] + subject  + os.sep + 'ses-*' + os.sep + 'dwi' + os.sep + '*.trk'
    trk_files = glob(trk_dir)
    
    logging.info(f'{len(trk_files)} TRK files found ')
    logging.info(trk_files)
    parallelize_CM(trk_files, config['tractogram_config']['cores'], run, config, general_dir)

if __name__ == '__main__':
    main()