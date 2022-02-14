# Spreading of misfolded proteins in Alzheimer’s Disease using Constrained Autoregressive Model

### Authors 
Luca Gherardini <br> 
Alessandro Crimi <br>
Aleksandra Pestka

### Overview 
Recent advancements in Magnetic Resonance Images acquisition have provided better estimations of brain properties through in vivo analysis, providing more additional data. At the same time, the better knowledge of Alzheimer's Disease biological mechanisms and progression allows the theorization of forecasting models on the clinical development of early-stage patients. We deployed the Constrained Multivariate Autoregressive Model (CMAR) with others already existing to provide an objective evaluation metric. Preprocessing steps applied cutting-edge optimization techniques to the data before using them in the models. We predicted the progression of the disease for each individual, using the structural connectivity extracted from MRIs and Positron Emission Tomography (PET) measures for Amyloid-Beta and Tau proteins. Experiments showed the reliability of CMAR in this kind of prediction, becoming a candidate for being an effective diagnostic tool.

### Methodology 
Simulations and models used for predicting th progression of Alzheimer disease:
- Multivariate Autoregressive Model (MAR)
- Network Diffusion Model (NDM)
- Epidemic Model Spreading (EMS)

### Workflow
1. Install dependencies.<br>
To install the necessary dependencies you can use ```pip install -r requirements.txt.``` It is advised to use it within a python3 virtual environment with Python3.9. 
2. Add atlas file and dataset (DWI + PET) to [data](data) folder according to the following structure:
```bash
├── ADNI
│   └── derivatives
│       ├── sub-AD4009
│       │   ├── ses-baseline
│       │   │   ├── dwi
│       │   │   └── pet
│       │   └── ses-followup
│       │       └── pet
│       └── sub-AD4215
│           ...
└── atlas
```
3. Generate tractography and connectivity matrix from DWI data. 
``` bash
cd src/tractography
python3 generate_tractogram_with_CM.py 
```
4. Extract average MP concentration in brain regions. 
```bash
cd src/analysis
python3 extract_regions_means_PET.py
```
6. Prepare dataset for training and testing. 
```bash
cd src/dataset_preparing
python3 create_dataset.py
```
7. Choose model for making predicitons (including training and testing):
```bash
cd src/simulations
python3 simulation_MAR.py 
python3 simulation_EMS.py
python3 simulation_NDM.py
```


### Project Organization
```bash
├── README.md                                      
├── requirements.txt                # requirement packages
├── config.yaml                     # configuration file for tractography and connectome generation
├── publications                    # reference publications
├── data
│   ├── ADNI                        # ADNI dataset 
│   └── atlas                       # brain atlas
├── app                             # basic web application
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
    ├── simulations                 # simulation scripts 
    │   ├── simulation_NDM.py 
    │   ├── simulation_EMS.py
    │   ├── simulation_MAR.py
    │   ├── utils.py
    │   └── utils_vis.py
    └── tractography                # tractography and connectomes generation 
        ├── generate_CM_v2.py
        ├── generate_connectivity_matrix.py
        ├── generate_tractogram_with_CM.py
        └── utils.py
```

### Citing this work
If you use this code, please cite this paper: TBA 