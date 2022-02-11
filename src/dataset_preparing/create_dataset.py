''' Prepare JSON file with paths to valid data:
Structure:
{
    'subj': {
        'conncetome': connectome_path,
        'baseline': baseline_path,
        'followup': followup_path
    }
}

Some conditions must be fullfilled:
1. followup must be 2 years after the baseline
2. sum(followup) > sum(baseline)
'''

import os
from glob import glob
import json
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)

def get_file_paths_for_subject(dataset_dir, subject, tracer='av45'):

    try:
        connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_rough.csv'))
    except Exception as e:
        logging.error(e)
        logging.error(f"Missing connectivity matrix for subject {subject}")
    
    try:
        t0_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'pet', f'*baseline*trc-{tracer}_pet.csv')))[0]
    except Exception as e:
        logging.error(e)
        logging.error(f"Error with t0 pet of patient {subject}")
    
    if not os.path.isfile(t0_concentration_path): f'No baseline for subject: {subject}'
    
    # extract baseline year 
    t0_year = int(t0_concentration_path.split('date-')[1][:4])

    # followup year should be: baseline year + time interval
    time_interval = 2

    # TODO: iterate over all the pet csv files of the patient and check it the current pet is 2 years after the previous one, otherwise reiter until the end of available pets
    try:
        t1_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                                           'ses-followup', 'pet', 
                                                           f'*{t0_year+time_interval}*trc-{tracer}_pet.csv')))[0]
    except Exception as e:
        logging.error(e)
        logging.error(f"Error with t1 pet of patient {subject}")
    
    results_dict = {
        "connectome": connectivity_matrix_path, 
        "baseline": t0_concentration_path, 
        "followup": t1_concentration_path
    }
    return results_dict

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def is_concentration_valid(paths):
    ''' Check if sum(t1 concentration) is greater than sum(t0 concentration). '''
    
    t0_concentration = load_matrix(paths['baseline']) 
    t1_concentration = load_matrix(paths['followup'])
                
    if sum(t1_concentration) > sum(t0_concentration):
        return True
    
    return False

def save_dataset(dataset, filename):
    with open(filename, 'w+') as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    dataset_filepath = 'dataset_av45.json'           
    dataset_dir = '../../data/ADNI/derivatives'
    
    subjects = os.listdir(dataset_dir)
    
    print(f'Initial no. of subjects: {len(subjects)}')
    
    dataset = {}
    for subj in subjects:
        try:
            paths = get_file_paths_for_subject(dataset_dir, subj) 
            if is_concentration_valid:
                dataset[subj] = paths
            else:
                logging.info(f'followup < baseline for: {paths["followup"]}')
        except IndexError:
            logging.info(f'No valid data for subject: {subj}')
            continue 
        
    save_dataset(dataset, dataset_filepath)
    logging.info(f'Size of the dataset: {len(dataset)}')