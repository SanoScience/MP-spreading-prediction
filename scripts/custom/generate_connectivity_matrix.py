''' Generate connectivity matrix from tractogram. '''

import numpy as np
from nibabel.streamlines import load
import nibabel
from dipy.tracking import utils 
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.measurements import label
matplotlib.use('TKAgg')
import os

# load tractogram data
path = '/home/bam/ADNI_2_3_BIDS_OPT/sub-AD1/ses-1/dwi/tractogram_sub-AD1_ses-1_acq-AP_dwi.nii.trk'
tractogram = load(path)

# tracks shorter than 30 mm are removed
longer_streamlines = [t for t in tractogram.streamlines if len(t)>30]

affine = tractogram.affine # transformation to align streamlines to atlas 

print(f'No. of streamlines: {np.shape(longer_streamlines)}')

# load atlas data
atlas = nibabel.load('/home/bam/Alexandra/Misfolded-protein-spreading/data/input/atlas/aal.nii.gz')
labels = atlas.get_fdata().astype(np.uint8)

print(np.unique(labels))
print(f'No. of unique atlas labels: {len(np.unique(labels))} mas value: {np.max(labels)}')

print('SHAPES: ', [np.shape(t) for t in [longer_streamlines, affine, labels]])

# create connectivity matrix; find out which regions of the brain are connected by these streamlines
M, grouping = utils.connectivity_matrix(longer_streamlines, 
                              affine=affine, 
                              label_volume=labels, 
                              return_mapping=True,
                              mapping_as_streamlines=True)

# remove background
M = M[1:, 1:]

print(f'Connectivity matrix shape: {M.shape}')

#Reshuffle making all left areas first right areas
odd_odd = M[::2, ::2]
odd_even = M[::2, 1::2]
print(odd_even.shape, odd_odd.shape)
first = np.vstack((odd_odd, odd_even))
even_odd = M[1::2, ::2]
even_even= M[1::2, 1::2]
second = np.vstack((even_odd, even_even))
M = np.hstack((first,second))

# remove connections to own regions (inplace)
np.fill_diagonal(M, 0)

# save connectivity matrix 
np.savetxt('/home/bam/ADNI_2_3_BIDS_OPT/sub-AD1/ses-1/dwi/connect_matrix.csv', M, delimiter=',')

# plot
plt.imshow(np.log1p(M), interpolation='nearest')
plt.savefig('/home/bam/ADNI_2_3_BIDS_OPT/sub-AD1/ses-1/dwi/connect_matrix.png')
