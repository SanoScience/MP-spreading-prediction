import numpy as np
import pandas as pd
import nibabel

COEFF_PATH = './data/MAR_model/A_matrix_MAR.csv' # path to MAR model coefficients matrix
ATLAS_PATH = './data/atlas/AAL3v1_1mm.nii.gz'    # path to brain atlas 

def load_coeff_matrix():
    data = np.genfromtxt(COEFF_PATH, delimiter=',')
    return data

def load_atlas():
    data = nibabel.load(ATLAS_PATH).get_fdata()
    return data

def drop_negative_predictions(baseline, prediction):
    ''' Get rid of negative values in predicted concentrations. 
    If negative value is encountered: replace it with the baseline values. '''
    return np.where(prediction<0, baseline, prediction)

def run_simulation(t0_concentration):
    ''' Predict MP concentration at time t1 based on data from time t0 '''
    coeff_matrix = load_coeff_matrix()
    t1_concentration_pred = coeff_matrix @ t0_concentration
    t1_concentration_pred = drop_negative_predictions(t0_concentration, t1_concentration_pred)
    return t1_concentration_pred

def calc_statistics(concentration):
    ''' Calculate some statistics from prediction ''' 
    stats = pd.DataFrame({'concentration': concentration}).describe().applymap(lambda x: f"{x:0.2f}")
    total_sum = np.round(np.sum(concentration), 2)
    top_brain_regions = np.argsort(concentration)[::-1][:5]
    return stats, total_sum, top_brain_regions

def prepare_atlas_with_concentrations(concentration):
    atlas = load_atlas()
    regions = np.unique(atlas).astype('int')
    
    # insert MP concentration = 0 for background and missing regions
    concentration = list(concentration)
    for idx in [0, 35, 36, 81, 82]: # missing regions labels
        concentration.insert(idx, 0)
    
    
    for region_idx in regions:
        atlas = np.where(atlas == region_idx, concentration[region_idx], atlas)
        
    # drop negative concentrations (especially in predictions)
    atlas = np.where(atlas < 0, 0, atlas)
    
    return atlas