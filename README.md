# Prediction of misfolded proteins spreading in Alzheimer’s disease using machine learning

### Authors 
Luca Gherardini <br> 
Alessandro Crimi <br>
Aleksandra Pestka


Simulations and models used for predicting th progression of Alzheimer disease:
- Constrained Multivariate Autoregressive Model (MAR)
- Network Diffusion Model (NDM)
- Epidemic Spreading Model (ESM)

### Workflow
1. Install dependencies.<br>
To install the necessary dependencies you can use ```pip install -r requirements.txt.``` It is advised to use it within a python3 virtual environment with Python3.9. 
2. Please, format your data according to the [BIDS format](https://bids.neuroimaging.io/):
```bash
├── ADNI
   └── derivatives
       ├── sub-AD4009
       │   ├── ses-baseline
       │   │   ├── dwi
       |   |   ├── anat
       │   │   └── pet
       │   └── ses-followup
       │       └── pet
       └── sub-AD4215
           ...

```
3. Perform preprocessing on your dataset
``` bash
cd src/preprocessing
python3 main.py <img_type> <dataset_path> <cores> <atlas_file>
```
*img_type*: type of image to look up for (dwi, anat or pet)
*dataset_path*: absolute or relative path of the input folder containing images (it can be '.' if its in the same path). Please note that the pipeline will produce a directory named 'derivatives' inside the specified path.
*cores*: the number of cores to use (-1 uses all available).
*atlas_file*: the path for the atlas to use during registration step. Insert 'anat' to coregister the current image to the corresponding anatomical T1-weighted file (if available).
The CLI options are facultative and will be asked before starting the preprocessing if not specified.

4. Generate tractography and connectivity matrix from DWI data. 
``` bash
cd src/tractography
python3 TRK_CM_generation.py
```
5. Extract average MP concentration in brain regions. 
```bash
cd src/analysis
python3 extract_regions_means_PET.py
```
6. Prepare dataset for training and testing. 
```bash
cd src/dataset_preparing
python3 create_dataset.py <cores> <threshold>
```
*threshold*: the tolerance (between 0 and 1) with which discarding a subject if the Amyloid-beta concentration at baseline is >= threshold * followup
7. Choose model for making predicitons (including training and testing):
```bash
cd src/simulations
python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <matrix> <use_binary> <iter_max> <N_fold>
```
*category*: clinical group on which perform simulations. A file in the dataset directory named dataset_<category>.json must exhist.
*lambda*: parameter in [0,1] used to set an L2 regression on the gradient descent optimization. Its utilization has not been considered in the papers results.
*matrix*: prior to use for matrix A (0: CM, 1: diag(baseline), 2: random, 3: identity)
*use_binary*: use binary matrix as constraint (1) or no (0)
```
python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise>
python3 simulation_NDM.py <category> <cores> <beta>
python3 simulation_GCN.py <category> <matrix> <epochs>
python3 simulation_CDRMR.py <category> <cores> <train_size> <iter_max> <N_fold>
```


### Project Organization
```bash
├── README.md                                      
├── requirements.txt                # requirement packages
├── config.yaml                     # configuration file for tractography and connectome generation
├── publications                    # reference publications
├── data
│   └── atlas                       # brain atlas
└── src                             # source code 
    ├── analysis                   
    │   ├── extract_regions_means_PET.py # extracting average MP concentration from brain regions
    │   ├── prediction_analysis.py
    │   ├── prediction_visualization.py
    │   ├── statistics_connectomes.py
    │   └── statistics_PET.py
    ├── dataset_preparing           # dataset preparation: input and output for MAR and simulations 
    │   ├── create_dataset.py
    │   ├── dataset_av45.json
    ├── preprocessing               # preprocessing scripts 
    │   ├── main.py
    │   └── utils
    │       ├── brain_extraction.py
    │       ├── brain_segmentation.py
    │       ├── cerebellum_normalization.py
    │       ├── denoising.py
    │       ├── eddy_correction.py
    │       ├── flatten.py
    │       ├── gibbs.py
    │       ├── motion_correction.py
    │       └── registration.py
    ├── simulations                 # simulation scripts 
    │   ├── simulation_NDM.py 
    │   ├── simulation_ESM.py
    │   ├── simulation_MAR.py
    │   ├── simulation_GCN.py
    │   ├── simulation_CDRMR.py
    │   ├── utils.py
    │   └── utils_vis.py
    └── tractography                # tractography and connectomes generation 
        ├── TRK_CM_generation.py
        └── utils.py
```

### Citing this work
If you use this code, please cite this paper: TBA 
