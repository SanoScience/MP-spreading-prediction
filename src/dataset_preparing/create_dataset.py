''' 
SYNOPSIS:
    python3 create_dataset.py <cores> <threshold>
    
Prepare JSON file with paths to valid data:
Structure:
{
    'subj': {
        'connectome': connectome_path,
        'baseline': baseline_path,
        'followup': followup_path
    }
}

Some conditions must be fullfilled:
1. followup must be 2 years after the baseline
2. sum(followup) > sum(baseline)
'''

import multiprocessing
import os
from glob import glob
import json
import logging

from collections import defaultdict
import numpy as np
import re
import yaml
from datetime import datetime
import sys
from spektral.data import Dataset, Graph

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

wrong_subjects = []

def compute_cdr(memory, orient, judge, community, home, care):
    '''
    The global CDR is derived from the scores in each of the six categories (“box scores”):
    Memory, Community Affairs, Orientation, Home and Hobbies, Judgment and Problem Solving, Personal Care
    Memory (M) is considered the primary category, the others are secondary.
    CDR	=	GLOBAL	BOX	SCORE
    M	=	MEMORY	BOX	SCORE
    CDR	=	M	IF	AT	LEAST	THREE	SECONDARY	CATEGORIES	ARE	GIVEN THE	SAME	SCORE	AS	
    MEMORY.
    1) When M = 0.5, CDR = 1 if at least three of the other categories are scored 1 or greater.
    2) If M = 0.5, CDR cannot be 0; it can only be 0.5 or 1.
    3) If M = 0, CDR = 0 unless there is impairment (0.5 or greater) in two or more secondary categories, in which case CDR = 0.5.
    4) Whenever three or more secondary categories are given a score greater or less than the memory score, 
        (4a) CDR = score of majority of secondary categories on whichever side of M has the greater number of secondary categories. 
        (4b) In the unusual circumstance in which three secondary categories are scored on one side of M and two secondary categories are scored on the other side of M, CDR = M. 
    
    The above rules do not cover all possible scoring combinations.
    Unusual circumstances are scored as follows:
    5) With ties in the secondary categories on one side of M, choose the tied scores closest
    to M for CDR (e.g. M and another secondary category = 3, two secondary categories =
    2, and two secondary categories = 1; CDR =2).
    6) When only one or two secondary categories are given the same score as M, CDR = M
    as long as no more than two secondary categories are on either side of M.
    7) When M = 1 or greater, CDR cannot be 0; in this circumstance, CDR = 0.5 when the
    majority of secondary categories are 0.
    '''
    
    # 0 (no scores stored, skip)
    if memory == -1 and orient ==-1 and judge == -1 and community == -1 and home == -1 and care == -1:
        return -1
        
    # 1
    if memory == .5:        
        if (orient >= 1) + (judge >= 1) + (community >= 1) + (home >= 1) + (care >= 1) >= 3:
            return 1
    
    # 3
    if memory == 0:
        if (orient >= .5) + (judge >= .5) + (community >= .5) + (home >= .5) + (care >= .5) >= 1:
            return 0.5
    
    # 7
    if memory >= 1 and ((orient == 0) + (judge == 0) + (community == 0) + (home == 0) + (care == 0)) >= 3:
        return 0.5
        
    # 4
    if (memory != orient) + (memory != judge) + (memory != community) + (memory != home) + (memory != care) >= 3:
        secondary_scores = np.array([orient, judge, community, home, care])
        scores, elected = np.unique(secondary_scores, return_counts=True)
        for i in range(len(scores)):
            # 4a: there is concordance
            if elected[i] == 3:
                return scores[i]
                break
        # 4b: there is not concordance
        else:
            return memory
    
    # 5
    return np.mean([memory, orient, judge, community, home, care])

def load_dict(subject, tracers= ['av45','fbb','pib']):    
    pets_list = []
    date_re = re.compile(r'\d{4}-\d{2}-\d{2}')
    try:
        connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_norm.csv'))
        if not os.path.isfile(connectivity_matrix_path): raise Exception(f"{connectivity_matrix_path} doesn't exist")

        for t in tracers:
            pets_list = pets_list + glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-*', 'pet', f'*trc-{t}_pet.csv')))
        #NOTE: year check has been manually done (see DEPRECATED below)
        for i in range(len(pets_list)):
            cdr = ''
            # find a CDR score close to t1 PET
            pet_date = datetime.strptime(date_re.findall(pets_list[i])[0], '%Y-%m-%d').date()
            for subj_cdr in cdrs:
                # the -1 skips the slash '/' (i.e. derivatives/sub-LMCI4712/)
                if subj_cdr == subject[-5:-1]:
                    for record in cdrs[subj_cdr]:
                        # accept CDR up to 2 years (~730 days) after/before t1 PET
                        days = abs(record[0] - pet_date).days
                        #days_threshold = 100 if 'ses-baseline' in pets_list[i] else 450
                        days_threshold = 730
                        if days < days_threshold:
                            if cdr != '' and days > abs(pet_date - cdr[0]).days:
                                # skip if the already existing CDR is closer to t1 PET than this one
                                continue
                            cdr = [record[0], record[1]]
                             
            if cdr == '':
                logging.error(f"{subject} has no CDR score for pet {pets_list[i]}")
                print(f"{subject} has no CDR score for pet {pets_list[i]}")
                cdr = ['n/a', 'n/a']
            if 'ses-baseline' in pets_list[i]:
                t0_concentration_path = pets_list[i]
                t0_concentration = load_matrix(t0_concentration_path) 
                cdr_t0 = cdr
            if 'ses-followup' in pets_list[i]:
                t1_concentration_path = pets_list[i]
                t1_concentration = load_matrix(t1_concentration_path)
                cdr_t1 = cdr        

        t0_sum = sum(t0_concentration)
        t1_sum = sum(t1_concentration)
        logging.info(f"Subject {subject} has t0={t0_sum} and t1={t1_sum}")
        if t1_sum < (t0_sum * threshold):
            wrong_subjects.append(subject)
            raise Exception(f"Subject {subject} has a gap baseline-followup of {t1_sum-t0_sum}")

        if t1_sum - t0_sum < 0:
            kind = 'Decreasing'
        else:
            kind = 'Increasing'
            
    except Exception as e:
        logging.error(f"Error while creating dataset for subject {subject}. Traceback: {e}")
        return   

    #graph_file = subject + subject.split('/')[-2] + '_graph.npz'
    #np.savez(graph_file, x=t0_concentration, y=t1_concentration, subject=subject)
    
    # NOTE: 'datetime' objects are converted to string to allow JSON serialization
    results_dict = {
    "CM": connectivity_matrix_path, 
    "baseline": t0_concentration_path, 
    "followup": t1_concentration_path,
    "CDR_t0_date": str(cdr_t0[0]),
    "CDR_t0_score": cdr_t0[1],
    "CDR_t1_date": str(cdr_t1[0]),
    "CDR_t1_score": cdr_t1[1],
    #"graph": graph_file,
    }
    
    datasets['ALL'].append([subject, results_dict])
    if kind == 'Increasing':
        datasets['Increasing'].append([subject, results_dict])
    else:
        datasets['Decreasing'].append([subject, results_dict])
    for c in categories:
        if re.match(rf".*sub-{c}.*", subject):
            datasets[c].append([subject, results_dict])        
    
    
    logging.info(f"Subject {subject} loaded")

    return    

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def save_dataset(dataset, filename):
    # Convert list to dictionary, using the first element (subject folder) as key
    ds = {subj[0]: subj[1] for subj in dataset}
    with open(filename, 'w+') as f:
        json.dump(ds, f, indent=4)

if __name__ == '__main__':
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.chdir(os.getcwd() + '/../../..') # Moving to the root folder of the project
    general_dir = os.getcwd() + os.sep
    logging.info(general_dir)
    dataset_dir = general_dir + config['paths']['dataset_dir'] + 'sub-*'  + os.sep
    logging.info(dataset_dir)
    subjects = glob(dataset_dir)

    dataset_output = config['paths']['dataset_dir'] + 'datasets/'
    if not os.path.isdir(dataset_output):
        os.mkdir(dataset_output)
    dataset_name = 'dataset_{}.json'          
    categories = ['ALL', 'AD', 'LMCI', 'MCI', 'EMCI', 'CN', 'Increasing', 'Decreasing']
    
    num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    while num_cores < 0:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
            if num_cores < 1: raise Exception("Invalid number of cores")
        except Exception as e:
            num_cores = multiprocessing.cpu_count()
        logging.info(f"{num_cores} cores available")

    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else -1
    while threshold < 0:
        try:
            threshold = float(input('Threshold from 0 to 1 [default 1]: '))
            if threshold < 0 or threshold > 1: raise Exception("Invalid number")
        except Exception as e:
            threshold = 1

    logging.info(f"Using threshold Followup >= {threshold} * Baseline")
        
    print(f'Initial no. of subjects: {len(subjects)}')
    
    logging.info(f"Reading CDR scores from \'CDR_compact.csv\'")
    print(f"Reading CDR scores from \'CDR_compact.csv\'")
    cdrs = defaultdict(list)
    # skip the header (first line)
    for line in open('CDR_compact.csv', 'r').readlines()[1:]:
        subj, date, memory, orient, judge, community, home, care, cdr_global = line.replace('\"', '').replace('\n', '').split(',')
        # filling the subject's ID with zeroes on the right to make it compatible with the number of digits of the ADNI archive for subject's ID
        subj = str.rjust(subj, 4, '0')
        
        memory = float(memory) if memory != '' else 0
        orient = float(orient) if orient != '' else 0
        judge = float(judge) if judge != '' else 0
        community = float(community) if community != '' else 0
        home = float(home) if home != '' else 0
        care = float(care) if care != '' else 0
        if cdr_global != '':
            cdr_global = float(cdr_global)
        else:
            cdr_global = compute_cdr(memory, orient, judge, community, home, care)
            logging.info(f'Imputed CDR score {cdr_global} for subject {subj}')
        
        # -1 is used to denote missing scores in the cdr archive
        if cdr_global == -1:
            continue
        
            
        date = datetime.strptime(date, '%Y-%m-%d').date()
        cdrs[subj].append([date, float(cdr_global)])
    
    datasets = {}
    for c in categories:
        datasets[c]= []
      
    logging.info('Reading PETs and creating datasets')
    print('Reading PETs and creating datasets')
    for subj in subjects:
        try:
            load_dict(subj)       
        except Exception:
            logging.error(f'No valid data for subject: {subj}')
            continue   
                
    for d in datasets.keys():
        save_dataset(datasets[d], dataset_output + dataset_name.format(d))
        logging.info(f'Size of the dataset \'{dataset_output + dataset_name.format(d)}\': {len(datasets[d])}')
        print(f'Size of the dataset \'{dataset_output + dataset_name.format(d)}\': {len(datasets[d])}')
    
    print(f"{len(wrong_subjects)} \'wrong\' subjects")
    logging.info(f"{len(wrong_subjects)} \'wrong\' subjects")
    logging.info(wrong_subjects)
