from nibabel import load, save, Nifti1Image
import numpy as np

atlas = load('AAL3v1_1mm.nii.gz')
atlas_data = atlas.get_fdata()
print(np.shape(atlas_data))
regions_labels = np.unique(atlas_data)[1:] # skip the first label (0, background)
print(np.shape(regions_labels))
print(regions_labels[:3], regions_labels[-3:])

regional_errors = np.loadtxt('NDM_ALL_regions_22-07-18_21:09:35.csv', delimiter=',')
print(regional_errors.shape)

for r in range(len(regions_labels)):
    atlas_data = np.where(atlas_data == regions_labels[r], regional_errors[r], atlas_data)

save(Nifti1Image(atlas_data, atlas.affine, atlas.header), 'NDM.nii.gz')