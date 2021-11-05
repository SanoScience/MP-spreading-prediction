''' Generate connectivity matrix from tractogram. '''

import os
import logging

import numpy as np
from nibabel.streamlines import load
import nibabel
from dipy.tracking import utils 
import matplotlib.pyplot as plt
import yaml

def get_paths(config):
    output_dir = os.path.join(config['paths']['output_dir'], 
                              config['paths']['subject'])
    atlas_path = config['paths']['atlas_path']
    tractogram_path = os.path.join(output_dir, 
                                   f"tractogram_{config['paths']['subject']}_ses-1_acq-AP_dwi_ACT.trk")
    return output_dir, atlas_path, tractogram_path

def remove_short_connections(streamlines, thres=30):
    longer_streamlines = [t for t in streamlines if len(t)>thres]
    return longer_streamlines

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

def create_connectivity_matrix(streamlines, affine, labels, reshuffle=True, take_log=True):
    ''' Get the no. of connections between each pair of brain regions. '''
    M, _ = utils.connectivity_matrix(streamlines, 
                                    affine=affine, 
                                    label_volume=labels, 
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
    if take_log: M = np.log1p(M)

    return M

def plot_connectivity_matrix(matrix, output_dir, take_log=True):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.title(f'Connectivity matrix (log values: {take_log})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connect_matrix.png'))

def main():
    logging.basicConfig(level=logging.INFO)

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    output_dir, atlas_path, tractogram_path =  get_paths(config)

    tractogram = load(tractogram_path)
    logging.info(f"Loading tractogram from subject: {config['paths']['subject']}")
    logging.info(f'No. of streamlines: {np.shape(tractogram.streamlines)}')

    affine = tractogram.affine # transformation to align streamlines to atlas 
    atlas, labels  = load_atlas(atlas_path)
    logging.info(f'No. of unique atlas labels: {len(np.unique(labels))}, \
        min value: {np.min(labels)}, max value: {np.max(labels)}')

    connect_matrix = create_connectivity_matrix(tractogram.streamlines, 
                                                affine, 
                                                labels,
                                                config['tractogram_config']['take_log'])
    
    np.savetxt(os.path.join(output_dir, 'connect_matrix.csv'), 
               connect_matrix, delimiter=',')
    logging.info(f'Shape of connectivity matrix: {connect_matrix.shape}. \
        Sum of values: {np.sum(connect_matrix)} (after removing background and connections to own regions)')

    plot_connectivity_matrix(connect_matrix, output_dir, 
                             config['tractogram_config']['take_log'])


if __name__ == '__main__':
    main()