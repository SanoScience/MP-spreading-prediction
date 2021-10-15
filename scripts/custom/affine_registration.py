''' 3D Image registration (Atlas and ADNI data).
Compute an affine transformation to register two 3D volumes 
by aligning the center of mass and other techniques. '''

import os

import numpy as np
import nibabel as nib
from dipy.viz import regtools
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import matplotlib.pyplot as plt


atlas_path = '../../data/input/atlas_reg.nii.gz'
adni_path = '../../data/input/sharepoint/ADNI/003_S_4136.nii.gz'
results_dir = '../../data/output/registration'

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# load 2 volumes (atlas - moving and ADNI data - static)
nib_adni = nib.load(adni_path)
static = np.squeeze(nib_adni.get_fdata())[..., 0]
static_grid2world = nib_adni.affine

nib_atlas = nib.load(atlas_path)
moving = np.array(nib_atlas.get_fdata())
moving_grid2world = nib_atlas.affine

print(f'ADNI image shape: {static.shape}, atlas image shape: {moving.shape}')

# plot overlaid slices from given volumes 
def save_overlaid_slices(img1, img2, filename, suptitle=False):
    # generate plot for every type of slice: {0:sagital, 1:coronal, 2:axial}
    for slice_type in [0, 1, 2]:
        fig = regtools.overlay_slices(img1, img2, None, slice_type, 
                                      'Static', 'Moving')
        if suptitle: fig.suptitle('Input images before alignment')
        plt.rcParams["figure.figsize"] = (20,3)
        plt.savefig(os.path.join(results_dir, 
                                 f'{filename}_{slice_type}.png'), dpi=600)

# transform the moving image using an identity transform 
identity = np.eye(4)
affine_map = AffineMap(identity,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
resampled = affine_map.transform(moving)
save_overlaid_slices(static, resampled, 'resampled')

# apply registration by aligning the centers of mass of two images
c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)
transformed = c_of_mass.transform(moving)
save_overlaid_slices(static, transformed, 'transformed_com')

# save transformed image in the NfTI format 
ni_img = nib.Nifti1Image(transformed, affine=np.eye(4))
nib.save(ni_img, os.path.join(results_dir, 'transformed.nii.gz'))

