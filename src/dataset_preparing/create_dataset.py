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
import queue
import threading

import numpy as np
import re
import yaml
from datetime import datetime
import sys

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

wrong_subjects = []

class datasetThread(threading.Thread):
   def __init__(self, threadID, dataset_dir, subject, threshold, queue, tracers= ['av45','fbb','pib']):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.dataset_dir = dataset_dir
        self.subject = subject
        self.queue = queue 
        self.tracers = tracers
        self.threshold = threshold

   def run(self):    
        pets_list = []
        try:
            connectivity_matrix_path = os.path.join(os.path.join(self.dataset_dir, self.subject, 
                                                'ses-baseline', 'dwi', 'connect_matrix_norm.csv'))
            if not os.path.isfile(connectivity_matrix_path): raise Exception(f"{connectivity_matrix_path} doesn't exist")

            for t in self.tracers:
                pets_list = pets_list + glob(os.path.join(os.path.join(self.dataset_dir, self.subject, 
                                                'ses-*', 'pet', f'*trc-{t}_pet.csv')))
            #NOTE: year check has been manually done (see DEPRECATED below)
            for i in range(len(pets_list)):
                if 'ses-baseline' in pets_list[i]:
                    t0_concentration_path = pets_list[i]
                    t0_concentration = load_matrix(t0_concentration_path) 
                if 'ses-followup' in pets_list[i]:
                    t1_concentration_path = pets_list[i]
                    t1_concentration = load_matrix(t1_concentration_path)

            t0_sum = sum(t0_concentration)
            t1_sum = sum(t1_concentration)
            logging.info(f"Subject {self.subject} has t0={t0_sum} and t1={t1_sum}")
            if t1_sum < (t0_sum*self.threshold):
                wrong_subjects.append(self.subject)
                raise Exception(f"Subject {self.subject} has a gap baseline-followup of {t1_sum-t0_sum}")

            if t1_sum - t0_sum < 0:
                kind = 'Decreasing'
            else:
                kind = 'Increasing'
                
        except Exception as e:
            logging.error(e)
            return None   

        results_dict = {
        "CM": connectivity_matrix_path, 
        "baseline": t0_concentration_path, 
        "followup": t1_concentration_path
        }
        
        self.queue.put([self.subject, results_dict, kind]) 
        
        logging.info(f"Subject {self.subject} loaded")

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
    datasets = {}
    for c in categories:
        datasets[c]= []
        
    queueLock = threading.Lock()
    dictQueue = queue.Queue(len(subjects))
    for subj in subjects:
        try:
            t = datasetThread(threading.active_count(), dataset_dir, subj, threshold, dictQueue)
            t.start()
            while threading.active_count() == num_cores+1:
                pass # simply wait
                    
        except Exception:
            logging.error(f'No valid data for subject: {subj}')
            continue 
        
    while threading.active_count() > 1:
        # wait for the termination of all threads (Note that one thread is the current main)
        pass

    while not dictQueue.empty():
        element = dictQueue.get()
        datasets['ALL'].append([element[0], element[1]])
        if element[2] == 'Increasing':
            datasets['Increasing'].append([element[0], element[1]])
        else:
            datasets['Decreasing'].append([element[0], element[1]])
        for c in categories:
            if re.match(rf".*sub-{c}.*", element[0]):
                datasets[c].append([element[0], element[1]])
                
        #dataset[element[0]] = element[1]
    for d in datasets.keys():
        save_dataset(datasets[d], dataset_output + dataset_name.format(d))
        logging.info(f'Size of the dataset \'{dataset_output + dataset_name.format(d)}\': {len(datasets[d])}')
        print(f'Size of the dataset \'{dataset_output + dataset_name.format(d)}\': {len(datasets[d])}')
    
    print(f"{len(wrong_subjects)} \'wrong\' subjects")
    logging.info(f"{len(wrong_subjects)} \'wrong\' subjects")
    logging.info(wrong_subjects)
