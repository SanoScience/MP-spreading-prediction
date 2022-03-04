''' Generate connectivity matrix from tractogram. '''

import os
import logging

import numpy as np
from nibabel.streamlines import load
import nibabel
from dipy.tracking import utils 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import take
import yaml
import re

def get_paths(config):
    sub = config['paths']['subject'] if config['paths']['subject'] != 'all' else 'sub-*'
    output_dir = os.path.join(config['paths']['output_dir'], 
                              )
    atlas_path = config['paths']['atlas_path']

    tractogram_path = [ t for t in os.walk(output_dir) if re.match(r"*.trk", t)] 
    return output_dir, atlas_path, tractogram_path

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

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

def main():
    logging.basicConfig(level=logging.INFO)

    # TODO: move working directory (os.chdir()) to be at the level of config file
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    os.chdir(os.getcwd() + '/../..')
    output_dir, atlas_path, tractogram_path =  get_paths(config)
    logging.info(f"Loading tractogram from subject: {config['paths']['subject']}")

    tractogram = load(tractogram_path)
    atlas, labels  = load_atlas(atlas_path)
    logging.info(f'No. of unique atlas labels: {len(np.unique(labels))}, \
        min value: {np.min(labels)}, max value: {np.max(labels)}')
    
    cm = ConnectivityMatrix(tractogram, labels, output_dir, 
                            config['tractogram_config']['take_log'])
    cm.process()

if __name__ == '__main__':
    main()