""" 
    SYNOPSIS
    python3 simulation_NDM.py <category> <cores> <beta> 
"""
''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import json
import multiprocessing
import os
import logging
from re import L
import sys
from time import time

from tqdm import tqdm 
import numpy as np
from scipy.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_error

from utils_vis import save_prediction_plot
from utils import drop_data_in_connect_matrix, load_matrix
from datetime import datetime
from prettytable import PrettyTable
import yaml
import networkx as nx

np.seterr(all = 'raise')
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_NDM_{date}.log")

class DiffusionSimulation:
    def __init__(self, connect_matrix, concentrations, beta):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations.
        t_total: gap between baseline and followup PET (in years)
            timestep = t_total / (iterations * 10)
            iterations = t_total / (timestep * 10)
            t_total = timestep * iterations * 10
        Putting iterations to '1' and for t_total = 2 years, the timestep is 0.2
        '''
        self.rois = 166 
        self.t_total = 2 # total length of the simulation in years
        self.timestep = self.t_total / 10
        self.cm = connect_matrix
        self.beta = beta
        if concentrations is not None: 
            #logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            #logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
        
        self.calc_laplacian()  

    '''
    DEPRECATED
    def define_seeds(self, init_concentration=1):
        # Define Alzheimer seed regions manually. 
        
        #Args:
        #    init_concentration (int): initial concentration of misfolded proteins in the seeds.
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (Cingulate gyrus, anterior/posterior part, left & right)
        # assign initial concentration of proteins in this region (please note index starts from 0, not from 1, then region 31 is at 30th index in the diffusion array)
        diffusion_init[[30, 31, 34, 35]] = init_concentration
        return diffusion_init
    '''

    def calc_laplacian(self, eps=1e-10): 
        self.cm = np.asmatrix(self.cm)
        G = nx.from_numpy_matrix(self.cm)
        self.L = nx.normalized_laplacian_matrix(G).toarray()
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        self.eigvecs = self.eigvecs.real
        self.eigvals = np.array(self.eigvals).real # Taking only the real part
        self.inv_eigvecs = np.linalg.inv(self.eigvecs + eps)
                    
    def run(self, downsample=False):
        ''' Run simulation. '''
   
        self.diffusion_final = self.iterate_spreading()

        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        return self.diffusion_final[-1]

    
    def integration_step(self, x_prev):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        step = 0
        try:
            exp = np.exp(self.beta * self.eigvals * -1)
            step = self.eigvecs * exp * self.eigvecs @  self.diffusion_init
        except Exception as e:
            logging.error(e)
        xt = x_prev + step * self.timestep
        return xt
    
    def iterate_spreading(self):
        diffusion = [self.diffusion_init]  
        
        try:
            next_step = self.integration_step(diffusion[-1])
        except Exception as e:
            logging.error(e)
            #break
        diffusion.append(next_step)  
            
        return np.asarray(diffusion, dtype=object) 
 
    def downsample_matrix(self, matrix, target_len=int(1e3)):
        ''' Take every n-th sample when the matrix is longer than target length. '''
        current_len = matrix.shape[0]
        if current_len > target_len:
            factor = int(current_len/target_len)
            matrix = matrix[::factor, :] # downsampling
        return matrix

def run_simulation(subject, paths, beta, queue=None):    
    ''' Run simulation for single patient. '''
      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['CM']))
        #connect_matrix = prepare_cm(connect_matrix)
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(f"Error during data load for subject {subject}. Traceback: ")
        logging.error(e)
        return

    try:
        simulation = DiffusionSimulation(connect_matrix, t0_concentration, beta)
    except Exception as e:
        logging.error(f"Error during simulation initialization for subject {subject}. Traceback: ")
        logging.error(e)
        return

    try:
        t1_concentration_pred = simulation.run()
    except Exception as e:
        logging.error(f"Error during simulation for subject {subject}. Traceback: ")
        logging.error(e)
        return
    
    try:
        mse = mean_squared_error(t1_concentration_pred, t1_concentration)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
        if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(f"Error during computation of statistics for subject {subject}. Traceback: ")
        logging.error(e)
        return
    
    logging.info(f"Saving prediction for subject {subj}")
    try:
        np.savetxt(os.path.join(subject, 'NDM_diffusion_' + date + '.csv'), simulation.diffusion_final, delimiter=',')
        np.savetxt(os.path.join(subject, 'NDM_terminal_concentrations_' + date + '.csv'), simulation.diffusion_final[-1, :], delimiter=',')
        save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subj, os.path.join(subject, 'NDM_' + date + '.png'), mse, corr_coef)
    except Exception as e:
        logging.error(f"Error during save of prediction for subject {subject}. Traceback: ")
        logging.error(e)
        return

    queue.put([subj, mse, corr_coef])
    
### MULTIPROCESSING ###

if __name__ == '__main__':
    total_time = time()
    
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.error(e)
            category = 'ALL'
        category = 'ALL' if category == '' else category
        
    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + 'simulations/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "MSE", "Pearson"]
    pt_subs.sortby = "ID" # Set the table always sorted by patient ID

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            num_cores = multiprocessing.cpu_count()
            logging.info(f"{num_cores} cores available")
            
    beta = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while beta < 0: 
        try:
            beta = float(input('Insert the beta value [0.1 by default]: '))
        except Exception as e:
            logging.error(e)
            beta = 0.1

    mse_list = []
    pcc_list = []
    
    procs = []
    queue = multiprocessing.Queue()
    for subj, paths in tqdm(dataset.items()):
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, beta, queue))
        p.start()
        procs.append(p)

        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                p.join(timeout=10)
                if not p.is_alive():
                    procs.remove(p)
        
    for p in procs:
        p.join()

    while not queue.empty():
        subj, err, pcc = queue.get()
        mse_list.append(err)
        pcc_list.append(pcc)
        pt_subs.add_row([subj, round(err,5), round(pcc,5)])

    pt_avg.add_row([round(np.mean(mse_list, axis=0), 5), round(np.std(mse_list, axis=0), 2), round(np.mean(pcc_list, axis=0), 5), round(np.std(pcc_list, axis=0), 2)])

    total_time = time() - total_time
    filename = f"{output_res}/NDM_{category}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
    out_file= open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Elapsed time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")