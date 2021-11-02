''' Generate connectivity matrix from tractogram. '''

import os

import numpy as np
from nibabel.streamlines import load
import nibabel
from dipy.tracking import utils 
import matplotlib.pyplot as plt
import yaml

def remove_short_connections(streamlines, thres=30):
    longer_streamlines = [t for t in streamlines if len(t)>thres]
    return longer_streamlines

def load_atlas(path):
    atlas = nibabel.load(path)
    labels = atlas.get_fdata().astype(np.uint8)
    return atlas, labels   

def create_connectivity_matrix(streamlines, affine, labels, reshuffle=True):
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

    return M

def plot_connectivity_matrix(matrix, output_dir, take_log=True):
    if take_log: matrix = np.log1p(matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.title(f'Connectivity matrix (log values: {take_log})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'connect_matrix.png'))

def main():
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    output_dir = os.path.join(config['paths']['output_dir'], 
                              config['paths']['subject'])
    atlas_path = config['paths']['atlas_path']
    
    tractogram = load(os.path.join(output_dir, 'tractogram_sub-AD1_ses-1_acq-AP_dwi.trk'))
    print(f'No. of streamlines: {np.shape(tractogram.streamlines)}')

    affine = tractogram.affine # transformation to align streamlines to atlas 
    atlas, labels  = load_atlas(atlas_path)
    print(f'No. of unique atlas labels: {len(np.unique(labels))}, \
        min value: {np.min(labels)}, max value: {np.max(labels)}')

    connect_matrix = create_connectivity_matrix(tractogram.streamlines, 
                                                affine, 
                                                labels)
    np.savetxt(os.path.join(output_dir, 'connect_matrix.csv'), 
               connect_matrix, delimiter=',')
    print(f'Shape of connectivity matrix: {connect_matrix.shape}. \
        Sum of values: {np.sum(connect_matrix)} (after removing background and connections to own regions)')

    plot_connectivity_matrix(connect_matrix, output_dir)


if __name__ == '__main__':
    main()