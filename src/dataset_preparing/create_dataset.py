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
import re

logging.basicConfig(level=logging.INFO)

def get_file_paths_for_subject(dataset_dir, subject, tracer='av45'):
    tracer= ['av45','fbb','pib']
    pets_list = []
    try:
        connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_rough.csv'))
        if not os.path.isfile(connectivity_matrix_path): raise Exception(f"{connectivity_matrix_path} doesn't exist")

        for t in tracer:
            pets_list = pets_list + glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-*', 'pet', f'*trc-{t}_pet.csv')))
        time_interval = 2
        # note: if 'tracer' is a list containing several tracers, the pairs can be made of heterogeneous tracers
        for i in range(len(pets_list)-1):
            for j in range(i+1, len(pets_list)):
                year = int(pets_list[i].split('date-')[1][:4])
                year_next = int(pets_list[j].split('date-')[1][:4])
                if year == year_next - time_interval:
                    t0_concentration_path = pets_list[i]
                    t1_concentration_path = pets_list[j]
                    break # this exits only the inner loop
            else:
                # this means that inner loop has been completed without break statements, so the outer loop must continue
                continue
            # if the previous continue statement has not been hit, it means the inner loop has been interrupted because a pair of valid PET has been found, then we can exit
            break
        else:
            # this 'else' means: 'if the for ended without finding a valid couple'
            raise Exception(f"{subject} doesn't have PET images with a {time_interval} years gap")
    except Exception as e:
        logging.error(e)
        return None   
    
    results_dict = {
        "connectome": connectivity_matrix_path, 
        "baseline": t0_concentration_path, 
        "followup": t1_concentration_path
    }
    return results_dict

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def is_concentration_valid(paths, threshold = 10):
    ''' Check if sum(t1 concentration) is greater than sum(t0 concentration). '''
    t0_concentration = load_matrix(paths['baseline']) 
    t1_concentration = load_matrix(paths['followup'])
                
    if sum(t1_concentration) > sum(t0_concentration) + threshold:
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
            if not is_concentration_valid(paths): raise Exception(f'followup < baseline for: {paths["followup"]}')
            if paths['connectome'] is None: raise Exception(f'connectivity matrix not found for {subj}')
            dataset[subj] = paths            
        except Exception:
            logging.error(f'No valid data for subject: {subj}')
            continue 
        
    save_dataset(dataset, dataset_filepath)
    logging.info(f'Size of the dataset: {len(dataset)}')