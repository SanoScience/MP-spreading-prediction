''' Generate connectivity matrix from tractogram. '''

import numpy as np
from nibabel.streamlines import load
import nibabel
from dipy.tracking import utils 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

# load tractogram data
path = '../../data/output/tractogram_003_S_4136.nii.trk'
tractogram = load(path)

# tracks shorter than 30 mm are removed
longer_streamlines = [t for t in tractogram.streamlines if len(t)>30]

affine = tractogram.affine # transformation to align streamlines to atlas 

print(f'No. of streamlines: {np.shape(longer_streamlines)}')

# load atlas data
atlas = nibabel.load('../../data/input/atlas_registered.nii.gz')
labels = atlas.get_fdata().astype(np.uint8)

print(f'No. of unique atlas labels: {len(np.unique(labels))}')

print('SHAPES: ', [np.shape(t) for t in [longer_streamlines, affine, labels]])

# create connectivity matrix; find out which regions of the brain are connected by these streamlines
M, grouping = utils.connectivity_matrix(longer_streamlines, 
                              affine=affine, 
                              label_volume=labels, 
                              return_mapping=True,
                              mapping_as_streamlines=True)

# remove background
M[:3, :] = 0
M[:, :3] = 0

# remove connections to own regions (inplace)
np.fill_diagonal(M, 0)

# save connectivity matrix 
np.savetxt('/home/bam/Misfolded-protein-spreading/data/output/connect_matrix.csv', M, delimiter=',')

# plot
plt.imshow(np.log1p(M), interpolation='nearest')
plt.savefig('../../data/output/connect_matrix.png')