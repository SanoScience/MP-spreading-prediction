''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

from cmath import inf, nan
import json
import multiprocessing
import os
import logging
from glob import glob
import random
from re import L
import sys
from time import time
from turtle import shape
import pandas as pd

from tqdm import tqdm 
import numpy as np
from scipy.stats.stats import pearsonr as pearson_corr_coef, PearsonRConstantInputWarning

from utils_vis import save_prediction_plot
from utils import drop_data_in_connect_matrix, load_matrix, calc_rmse, calc_rmse, prepare_cm
from datetime import datetime
from prettytable import PrettyTable

import networkx as nx
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)

class DiffusionSimulation:
    def __init__(self, connect_matrix, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations.
        t_total: gap between baseline and followup PET (in years)
            timestep = t_total / (iterations * 10)
            iterations = t_total / (timestep * 10)
            t_total = timestep * iterations * 10
        Putting iterations to '1' and for t_total = 2 years, the timestep is 0.02
        '''
        self.rois = 166 
        self.t_total = 2 # total length of the simulation in years
        self.timestep = self.t_total / 10 
        self.cm = connect_matrix
        if concentrations is not None: 
            #logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            #logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()

    def define_seeds(self, init_concentration=1):
        ''' Define Alzheimer seed regions manually. 
        
        Args:
            init_concentration (int): initial concentration of misfolded proteins in the seeds. '''
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (Cingulate gyrus, anterior/posterior part, left & right)
        # assign initial concentration of proteins in this region (please note index starts from 0, not from 1, then region 31 is at 30th index in the diffusion array)
        diffusion_init[[30, 31, 34, 35]] = init_concentration
        return diffusion_init

    def calc_laplacian(self, eps=1e-10): 
        self.cm = np.asmatrix(self.cm)
        G = nx.from_numpy_matrix(self.cm)
        self.L = nx.normalized_laplacian_matrix(G).toarray()
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        self.eigvecs = self.eigvecs.real
        self.eigvals = np.array(self.eigvals).real # Taking only the real part
        self.inv_eigvecs = np.linalg.inv(self.eigvecs + 1e-12)
                    
    def run(self, downsample=False):
        ''' Run simulation. '''

        # if values in the connectivity matrix were obtained through logarithm, revert it with an exponential 
        try:

            self.calc_laplacian()     
            # NOTE: this beta value has already been estimated and was the optimal for the considered dataset
            self.beta = 0.1
            self.diffusion_final = self.iterate_spreading()

            if downsample: 
                self.diffusion_final = self.downsample_matrix(self.diffusion_final)
            return self.diffusion_final[-1]

        except Exception as e:
            logging.error(e)

    
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
    
    def save_diffusion_matrix(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'diffusion_matrix_over_time.csv'), 
                                self.diffusion_final, delimiter=",")
    
    def save_terminal_concentration(self, save_dir):
        ''' Save last (terminal) concentration. '''
        np.savetxt(os.path.join(save_dir, 'terminal_concentration.csv'),
                   self.diffusion_final[-1, :], delimiter=',')

def run_simulation(subject, paths, output_dir, queue=None):    
    ''' Run simulation for single patient. '''

    subject_output_dir = os.path.join(output_dir, subject)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
        connect_matrix = prepare_cm(connect_matrix)
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        return

    try:
        simulation = DiffusionSimulation(connect_matrix, t0_concentration)
        t1_concentration_pred = simulation.run()
        rmse = calc_rmse(t1_concentration_pred, t1_concentration)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(rmse) or np.isinf(rmse): raise Exception("Invalid value of RMSE")
        if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(e)
        return
    
    save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subj, os.path.join(subject_output_dir, 'NDM_prediction.png'), rmse, corr_coef)

    if queue:
        queue.put([subj, rmse, corr_coef])
    
### MULTIPROCESSING ###

if __name__ == '__main__':
    total_time = time()

    os.chdir(os.getcwd()+'/../../')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.error(e)
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path = f'src/dataset_preparing/dataset_{category}.json'
    output_subj = 'results/subjects'
    output_res = 'results/benchmarks'
    if not os.path.exists(output_res):
        os.makedirs(output_res)

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg RMSE", "SD RMSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "RMSE", "Pearson"]
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

    
    rmse_list = []
    pcc_list = []
    
    # Testing (use the learned 'avg_beta')
    procs = []
    queue = multiprocessing.Queue()
    for subj, paths in dataset.items():
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_subj, queue))
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
        rmse_list.append(err)
        pcc_list.append(pcc)
        pt_subs.add_row([subj, round(err,2), round(pcc,2)])

    pt_avg.add_row([format(np.mean(rmse_list, axis=0), '.2f'), format(np.std(rmse_list, axis=0), '.2f'), format(np.mean(pcc_list, axis=0), '.2f'), format(np.std(pcc_list, axis=0), '.2f')])

    total_time = time() - total_time
    filename = f"{output_res}/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_NDM_{category}.txt"
    out_file= open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Elapsed time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")