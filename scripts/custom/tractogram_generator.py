import os
from pathlib import Path

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt

# define paths
fimg = "/home/bam/ADNI_2_3_BIDS_OPT/sub-AD2/ses-1/dwi/sub-AD2_ses-1_acq-AP_dwi.nii.gz"
fbval = "/home/bam/ADNI_2_3_BIDS_OPT/sub-AD2/ses-1/dwi/sub-AD2_ses-1_acq-AP_dwi.bval"
fbvec = "/home/bam/ADNI_2_3_BIDS_OPT/sub-AD2/ses-1/dwi/sub-AD2_ses-1_acq-AP_dwi.bvec"
output_dir = "/home/bam/ADNI_2_3_BIDS_OPT/sub-AD2/ses-1/dwi/"

data, affine, hardi_img = load_nifti(fimg, return_img=True) 
mask, binary_mask = median_otsu(data[:, :, :, 0]) 

# read file with diffusion weigths and orientation 
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

gtab = gradient_table(bvals, bvecs)

seed_mask = binary_mask # we want to seed from the whole brain, so use entire binary mask 
white_matter  = mask # TODO: should the white matter be the whole binary mask?
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
print(f'Init seeds: {seeds}, shape: {seeds.shape}')

# original fa_thr=0.7
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.01)

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)  
csd_fit = csd_model.fit(data, mask=white_matter)

csa_model = CsaOdfModel(gtab, sh_order=6) 
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
streamline_generator = LocalTracking(detmax_dg, stopping_criterion, 
                                     affine=affine,
                                     seeds = seeds,
                                     max_cross=1,
                                     step_size=.5,
                                     return_all=False)
streamlines = Streamlines(streamline_generator)

# generate and save tractogram 
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, os.path.join(output_dir, f"tractogram_{Path(fimg).stem}.trk"))

print("### DONE ###")

duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))