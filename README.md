# Misfolded proteins spreading 

Simulations part:
1) Diffusion model (using heat kernel)
2) Epidemic Model Spreading
3) Multivariate Autoregressive Model

Current **data** folder structure:

```
data/
├── ADNI
│   └── derivatives
│       ├── sub-AD4009
│       │   ├── ses-baseline
│       │   │   ├── anat
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1_mask.nii
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1.nii
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1_pve-0.nii
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1_pve-1.nii
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1_pve-2.nii
│       │   │   │   └── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_t1_seg.nii
│       │   │   ├── dwi
│       │   │   │   ├── connect_matrix.png
│       │   │   │   ├── connect_matrix_reverted.csv
│       │   │   │   ├── connect_matrix_rough.csv
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi_acqparams.txt
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi.bval
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi.bvec
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi_index.txt
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi.json
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi_mask.nii
│       │   │   │   ├── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi.nii
│       │   │   │   └── sub-AD4009_ses-baseline_acq-AP_date-2011-06-18_dwi_sc-act.trk
│       │   │   └── pet
│       │   │       ├── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet_cb-mask.nii
│       │   │       ├── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet_cb-only.nii
│       │   │       ├── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet.csv
│       │   │       ├── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet.json
│       │   │       ├── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet_mask.nii
│       │   │       └── sub-AD4009_ses-baseline_acq-AP_date-2011-07-07_trc-av45_pet.nii
│       │   └── ses-followup
│       │       └── pet
│       │           ├── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet_cb-mask.nii
│       │           ├── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet_cb-only.nii
│       │           ├── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet.csv
│       │           ├── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet.json
│       │           ├── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet_mask.nii
│       │           └── sub-AD4009_ses-followup_acq-AP_date-2013-07-03_trc-av45_pet.nii
│       └── sub-AD4215
│           └── ...
│
└── atlas
    └── aal.nii.gz
```