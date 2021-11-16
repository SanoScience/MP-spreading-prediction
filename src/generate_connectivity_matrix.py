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

def get_paths(config):
    output_dir = os.path.join(config['paths']['output_dir'], 
                              config['paths']['subject'])
    atlas_path = config['paths']['atlas_path']
    tractogram_path = os.path.join(output_dir, 
                                   f"tractogram_{config['paths']['subject']}_ses-1_acq-AP_dwi_ACT.trk")
    return output_dir, atlas_path, tractogram_path

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

class ConnectivityMatrix():
    def __init__(self, tractogram, atlas_labels, output_dir, take_log):
        self.streamlines = tractogram.streamlines
        self.affine = tractogram.affine # transformation to align streamlines to atlas 
        self.labels = atlas_labels  
        self.output_dir = output_dir
        self.take_log = take_log
                       
    def __create(self, reshuffle=True):
        ''' Get the no. of connections between each pair of brain regions. '''
        M, _ = utils.connectivity_matrix(self.streamlines, 
                                        affine=self.affine, 
                                        label_volume=self.labels, 
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
        # remove background
        M = M[1:, 1:]

        if reshuffle:
            # make all left areas first 
            odd_odd = M[::2, ::2]
            odd_even = M[::2, 1::2]
            first = np.vstack((odd_odd, odd_even))
            even_odd = M[1::2, ::2]
            even_even= M[1::2, 1::2]
            second = np.vstack((even_odd, even_even))
            M = np.hstack((first,second))

        # remove connections to own regions (inplace)
        np.fill_diagonal(M, 0) 
        if self.take_log: M = np.log1p(M)

        self.matrix = M
        
    def __save(self):
        np.savetxt(os.path.join(self.output_dir, 'connect_matrix.csv'), 
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
        
    def process(self):
        self.__create()   
        self.__get_info()
        self.__save()
        self.__plot()     

def main():
    logging.basicConfig(level=logging.INFO)

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
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