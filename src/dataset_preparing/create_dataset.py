''' Prepare JSON file with paths to valid data:
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

            """
            WARNING
            Higher levels of Amyloid-Beta at followup are not sustained by scientific evidence, and the followup could present lower concentrations than baseline due to scanner/computational defects. We reject only pets with very negative changes in deposition levels
            Read "PET amyloid-beta imaging in preclinical Alzheimer's disease" by Vlassenko, Andrei G. and Benzinger, Tammie L.S. and Morris, John C.
            if sum(t1_concentration) <= (sum(t0_concentration) + self.threshold):
                wrong_subjects.append(self.subject)
                raise Exception(f"{self.subject} PET images ({t0_concentration_path} and {t1_concentration_path}) don't have a concentration gap greater than {self.threshold}")
            """
        except Exception as e:
            logging.error(e)
            return None   

        results_dict = {
        "CM": connectivity_matrix_path, 
        "baseline": t0_concentration_path, 
        "followup": t1_concentration_path
        }
        # Assuring consistency while accessing dictionary
        queueLock.acquire()
        self.queue.put([self.subject, results_dict]) 
        queueLock.release()  
        logging.info(f"Subject {self.subject} loaded")

        """
        DEPRECATED
        time_interval = 2
            # note: if 'tracer' is a list containing several tracers, the pairs can be made of heterogeneous tracers
            # note: I need to compare pets in both orders, because I can't assume the retrieved list is in chronological order
            for i in range(len(pets_list)):
                for j in range(len(pets_list)):
                    if i == j:
                        continue

                    year = int(pets_list[i].split('date-')[1][:4])
                    year_next = int(pets_list[j].split('date-')[1][:4])
                    if year == year_next - time_interval:
                        t0_concentration_path = pets_list[i]
                        t1_concentration_path = pets_list[j]
                        t0_concentration = load_matrix(t0_concentration_path) 
                        t1_concentration = load_matrix(t1_concentration_path)
                        if sum(t1_concentration) >= (sum(t0_concentration) + self.threshold):
                            break # this exits only the inner loop
                        elif year < year_next and sum(t1_concentration) < sum(t0_concentration):
                            wrong_subjects.append(t1_concentration_path)
                else:
                    # this means that inner loop has been completed without break statements, so the outer loop must continue
                    continue
                # if the previous continue statement has not been hit, it means the inner loop has been interrupted because a pair of valid PET has been found, then we can exit
                break
            else:
                # this 'else' means: 'if the for ended without finding a valid couple'
                raise Exception(f"{self.subject} doesn't have PET images with a {time_interval} years gap and/or a concentration gap greater than {self.threshold}")
        except Exception as e:
            logging.error(e)
            return None   
        """   

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
    categories = ['ALL', 'AD', 'LMCI', 'MCI', 'EMCI', 'CN']
    
    num_cores = ''
    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        if num_cores < 1: raise Exception("Invalid number of cores")
    except Exception as e:
        num_cores = multiprocessing.cpu_count()
    logging.info(f"{num_cores} cores available")

    threshold = ''
    try:
        threshold = float(input('Threshold from 0 to 1 [default 1]: '))
        if threshold < 0 or threshold > 1: raise Exception("Invalid number")
    except Exception as e:
        threshold = 1
    
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
        for c in categories:
            if re.match(rf".*sub-{c}.*", element[0]):
                datasets[c].append([element[0], element[1]])
                
        #dataset[element[0]] = element[1]
    for d in datasets.keys():
        save_dataset(datasets[d], dataset_output + dataset_name.format(d))
        logging.info(f'Size of the dataset \'{dataset_output + dataset_name.format(d)}\': {len(datasets[d])}')
    logging.info(f"{len(wrong_subjects)} \'wrong\' subjects")
    logging.info(wrong_subjects)