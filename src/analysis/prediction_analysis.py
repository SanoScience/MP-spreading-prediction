import os
from glob import glob
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def normalization(data):
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized

def find_regions_above_mean(data):
    mean = np.mean(data)
    logging.info(f'Mean: {mean:.3f}')
    args = np.argwhere(data > mean)
    return args.flatten()

def main():
    output_dir = '../../results'                                                
    concentrations_dir = '../../data/PET_regions_concentrations'
    
    patients = ['sub-AD4215', 'sub-AD4009']
    for subject in patients:
        logging.info(f'\nSimulation for subject: {subject}')
        
        # load ground-truth t1 concentration
        true_concentrations_paths = glob(os.path.join(concentrations_dir, subject, '*.csv'))                               
        t1_concentration_path = [path for path in true_concentrations_paths if 'followup' in path][0]            
        t1_concentration = load_matrix(t1_concentration_path)
        
        # load predicted t1 concentration
        pred_concentrations_path = os.path.join(output_dir, subject, 'concentration_pred_EMS.csv')
        t1_concentration_pred = load_matrix(pred_concentrations_path)
        
        # do some analysis: normalization and mean calculation
        t1_concentration_pred_norm = normalization(t1_concentration_pred)
        above_mean_regs = find_regions_above_mean(t1_concentration_pred_norm)
        print(f'Index of regions above the mean: {above_mean_regs}')
        

if __name__ == '__main__':
    main()