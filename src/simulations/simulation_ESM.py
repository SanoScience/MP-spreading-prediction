"""
    SYNOPSIS
    python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise>
"""

''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

Based on publication: 
"Epidemic Spreading Model to Characterize Misfolded Proteins Propagation 
in Aging and Associated Neurodegenerative Disorders"
Authors: Yasser Iturria-Medina ,Roberto C. Sotero,Paule J. Toussaint,Alan C. Evans 
'''

from datetime import datetime
import json
from threading import Thread, Lock
from multiprocessing import cpu_count
import os
import logging
import sys
from time import time, sleep
import warnings
from prettytable import PrettyTable
import yaml
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_error
import re

from utils_vis import *
from utils import *

np.seterr(all = 'raise')
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_ESM_{date}.log")
digits = 4

class ESM(Thread):       
    
    def __init__(self, paths, subj, n_regions = 166):
        Thread.__init__(self)
        self.paths = paths 
        self.subj = subj
        self.n_regions = n_regions

    def drop_negative_predictions(self, predictions):
        return np.maximum(predictions, 0)

    def compute_gini(self):
        '''
        https://en.wikipedia.org/wiki/Gini_coefficient#Definition
        Compute Gini coefficient as half of the relative mean absolute difference, which is mathematically equivalent to the definition based on the Lorenz curve.
        The mean absolute difference is the average absolute difference of all pairs of items of the population, and the relative mean absolute difference is the mean absolute difference divided by the average to normalize for scale.
        '''
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(self.t0, self.t0)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(self.t0)
        # Gini coefficient
        return 0.5 * rmad

    def simulation(self):
        
        #TODO: should I use something different from 0 for null concentrations? (i.e. 0.1) Possibly depending on strenght of connections
        # between healthy and infected regions (and also internal concentrations)
        try:        
            #P = define_initial_P(concentration, self.cm, distances, max_concentration)
            # NOTE Initial P has to be explored for different values (i.e. 0.5 instead of 1)
            P = np.array(np.where(self.t0>0, 1, 0), dtype=np.float64)
            iterations = 2 # found through empirical trials
            timestep = 1e-2
        except Exception as e:
            logging.error(f"Error while initializing variables: {e}")
            return self.t0
        
        for k in range(iterations):
        
            with warnings.catch_warnings():
                warnings.filterwarnings ('error')
                try:
                    # Compute Beta and Delta vectors
                    Beta = 1 - np.exp(- beta_0 * P) # Diffusion
                    Delta = np.exp(- delta_0 * P) # Recovery
                    gini = self.compute_gini()
                    
                    Beta_ext = gini * Beta 
                    Beta_int = (1 - gini) * Beta
                    
                    # Epsilon has to be computed at each time step
                    Epsilon = np.zeros(self.n_regions)
                    # Define gaussian noise (NOTE authors don't differentiate it by region nor by time)
                    noise = np.random.normal(mu_noise, sigma_noise)
                except Exception as e: 
                    logging.error(f"Iteration {k}: Error in computing Beta, Delta, Epsilon, gini or noise. Traceback: {e}")
                    return self.t0
                try:
                    for i in range(self.n_regions):
                        for j in range(self.n_regions):
                            if i != j:
                                # NOTE: I am assuming the delay to go from j to i is null (else Beta_ext(t- Tau(ij)))
                                # Computing the extrinsic infection probability (i!=j)...
                                Epsilon[i] += self.cm[j,i] * Beta_ext[j] * P[i] 
                        # and summing the intrinstic infection rate
                        Epsilon[i] += self.cm[i,i] * Beta_int[i] * P[i]
                
                    # Updating probabilities
                    P += ((1 - P) * Epsilon - Delta * P + noise) * timestep
                    assert P.all()<=1 and P.all()>=0, "Probabilities are not between 0 and 1"
                except Exception as e:
                    logging.error(f"Iteration {k}: Error in updating P. Traceback: {e}")
                    return self.t0
            
            try:
                self.t0 += (P * (self.t0 @ self.cm)) * timestep
            except Exception as e:
                logging.error(f"Iteration {k}: Error in updating concentration. Traceback: {e}")
                return self.t0        
            
        return self.t0
        
    def run(self): 
        logging.info(f"Starting simulation for subject {self.subj}")
        if not os.path.exists(self.subj + 'test/'):
                os.makedirs(self.subj + 'test/')
        try:
            self.cm = drop_data_in_connect_matrix(load_matrix(self.paths['CM']))
            # ESM uses self connections for inner propagation (CM has a diagonal set to 0 due to normalization during CM generation)
            self.cm += np.identity(self.cm.shape[0])
            self.t0_concentration = load_matrix(self.paths['baseline'])
            self.t0 = np.copy(self.t0_concentration)
            self.t1_concentration = load_matrix(self.paths['followup'])
        except Exception as e:
            logging.error(f'Error appening while loading data of subject {self.subj}. Traceback: {e}')
            return

        try:
            self.t1_concentration_pred = self.simulation()

            self.t1_concentration_pred = self.drop_negative_predictions(self.t1_concentration_pred)
            if np.isnan(self.t1_concentration_pred).any() or np.isinf(self.t1_concentration_pred).any(): raise Exception("Discarding prediction")
        except Exception as e:
            logging.error(f'Error during simulation for subject {self.subj}. Traceback: {e}')
            return
        
        try:
            mse = mean_squared_error(self.t1_concentration, self.t1_concentration_pred)
            pcc = pearson_corr_coef(self.t1_concentration, self.t1_concentration_pred)[0]
            if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
        except Exception as e:
            logging.error(f'Error appening during computation of MSE and PCC for subject {self.subj}. Traceback: {e}')
            return
        
        
        reg_err = np.abs(self.t1_concentration_pred - self.t1_concentration)
        
        lock.acquire()
        save_prediction_plot(self.t0_concentration, self.t1_concentration_pred, self.t1_concentration, self.subj, self.subj + 'test/ESM_' + date + '.png', mse, pcc)
        logging.info(f"Saving prediction in {self.subj + 'test/ESM_' + date + '.png'}")
        total_mse[self.subj] = mse
        total_pcc[self.subj] = pcc
        total_reg_err[self.subj] = reg_err
        pt_subs.add_row([self.subj, round(mse,digits), round(pcc,digits)])
        lock.release()
            
        return

if __name__=="__main__":

    ### INPUT ###

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if os.getcwd().endswith('simulations'):
        os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.info("Using default value")
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)
    if not os.path.exists(output_mat):
        os.makedirs(output_mat)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            num_cores = cpu_count()
            logging.info(f"{num_cores} cores available")
            
    beta_0 = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while beta_0 < 0:
        try:
            beta_0 = float(input('Insert the value for beta_0 [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            beta_0 = 0
    
    delta_0 = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while delta_0 < 0:
        try:
            delta_0 = float(input('Insert the value for delta_0 [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            delta_0 = 0
    
    mu_noise = float(sys.argv[5]) if len(sys.argv) > 5 else -1
    while mu_noise < 0:
        try:
            mu_noise = float(input('Insert the value for mu_noise [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            mu_noise = 0
            
    sigma_noise = float(sys.argv[6]) if len(sys.argv) > 6 else -1
    while sigma_noise < 0:
        try:
            sigma_noise = float(input('Insert the value for sigma_noise [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            sigma_noise = 0
    
    ### SIMULATIONS ###

    pt_avg = PrettyTable()
    pt_avg.field_names = ["CG", "Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "MSE", "Pearson"]
    pt_subs.sortby = "ID" # Set the table always sorted by patient ID

    total_mse = {}
    total_pcc = {}
    total_reg_err = {}
    
    total_time = time()

    lock = Lock()
    
    works = []
    for subj, paths in tqdm(dataset.items()):
        works.append(ESM(paths, subj))
        works[-1].start()
        while len (works) >= num_cores:
            for w in works:
                if not w.is_alive():
                    works.remove(w)
            
    for w in works:
        w.join()
        works.remove(w)
        
        
    total_time = time() - total_time
    sleep(1)   
    ### OUTPUT ###
    categories = ['AD', 'LMCI', 'MCI', 'EMCI', 'CN', 'Decreasing', 'Increasing']
    
    
    for c in categories:
        cat_reg_err = []
        cat_total_mse = []
        cat_total_pcc = []
        for sub in total_reg_err.keys():
            if re.match(rf".*sub-{c}.*", sub):
                cat_reg_err.append(total_reg_err[sub])
                cat_total_mse.append(total_mse[sub])
                cat_total_pcc.append(total_pcc[sub])

        if len(cat_reg_err) == 0:
            continue
        avg_reg_err = np.mean(cat_reg_err, axis=0)
        avg_reg_err_filename = output_res +f'ESM_region_{c}_{date}.png'
        save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
        np.savetxt(f"{output_mat}ESM_{c}_regions_{date}.csv", avg_reg_err, delimiter=',')
        avg_mse = np.mean(cat_total_mse, axis=0)
        std_mse = np.std(cat_total_mse, axis=0)
        avg_pcc = np.mean(cat_total_pcc, axis=0)
        std_pcc = np.std(cat_total_pcc, axis=0)
        
        pt_avg.add_row([c, round(avg_mse, digits), round(std_mse, 2), round(avg_pcc, digits), round(std_pcc, 2)])
    
    if category not in categories:
        pt_avg.add_row([category, round(np.mean(list(total_mse.values())), digits), round(np.std(list(total_mse.values())), 2), round(np.mean(list(total_pcc.values())), digits), round(np.std(list(total_pcc.values())), 2)])
        avg_reg_err = np.mean(list(total_reg_err.values()), axis=0)
        avg_reg_err_filename = output_res +f'ESM_region_{category}_{date}.png'
        save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
        np.savetxt(f"{output_mat}ESM_{category}_regions_{date}.csv", avg_reg_err, delimiter=',')

    filename = f"{output_res}ESM_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Beta_0: {beta_0}\n")
    out_file.write(f"Delta_0: {delta_0}\n")
    out_file.write(f"mu_noise: {mu_noise}\n")
    out_file.write(f"sigma_noise: {sigma_noise}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())    
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Beta_0: {beta_0}")
    logging.info(f"Delta_0: {delta_0}")
    logging.info(f"mu_noise: {mu_noise}")
    logging.info(f"sigma_noise: {sigma_noise}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")