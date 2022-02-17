''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import json
import multiprocessing
import os
import logging
from glob import glob
import random
import sys
from time import time
from turtle import shape
import pandas as pd

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian
from scipy.stats.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_log_error

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import drop_data_in_connect_matrix, load_matrix, calc_rmse, calc_rmse
from datetime import datetime
from prettytable import PrettyTable

import networkx as nx

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.DEBUG)

class DiffusionSimulation:
    def __init__(self, connect_matrix, beta, timestep, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.beta = beta 
        self.rois = 166 
        self.t_total = 2 # total length of the simulation in years
        self.timestep = timestep # equivalent to 7.3 days per time step
        self.iterations = int(self.t_total / self.timestep) # 200 iterations
        self.cm = connect_matrix
        if concentrations is not None: 
            #logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            #logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
                    
    def run(self, inverse_log=True, downsample=False):
        ''' Run simulation. '''

        # if values in the connectivity matrix were obtained through logarithm, revert it with an exponential 
        try:
            if inverse_log: self.calc_exponent()

            self.calc_laplacian()        
            self.diffusion_final = self.iterate_spreading_by_Julien()

            # self.diffusion_final = self.iterate_spreading()
            if downsample: 
                self.diffusion_final = self.downsample_matrix(self.diffusion_final)
            return self.diffusion_final[-1]

        except Exception as e:
            print(e)
        
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
        # Laplacian: L = D - A
        # assume: A - adjacency matrix, D - degree matrix, I - identity matrix, L - laplacian matrix
        self.cm = np.asmatrix(self.cm)
        G = nx.from_numpy_matrix(self.cm)
        # normalized Laplacian: L = I - D-1/2 @ A @ D-1/2
        self.L = nx.normalized_laplacian_matrix(G).toarray()
        
        # this is not the degree matrix
        #D = np.diag(np.sum(A, axis=1))# total no. of. connections to other vertices
        #I = np.identity(A.shape[0]) # identity matrix
        #D_inv_sqrt = np.linalg.inv(np.sqrt(D)+eps) # add epsilon to avoid getting 0 determinant
        #self.L = I - (D_inv_sqrt @ A) @ D_inv_sqrt           

        # eigendecomposition
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
    
    def integration_step_by_Julien(self, x_prev, timestep):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        # x(t)/dt = -B * H * x(t)
        #xt = x_prev - timestep * self.beta * self.L @ x_prev
        
        # where x(t) = e^(-B*H*t) * x0
        #xt = x_prev - timestep * self.beta * self.L @ x_prev

        # where x(t) = U * e^(-lambda*B*t) * U^(-1) * x0
        step = self.eigvecs @ np.exp(-self.eigvals * self.beta * timestep) @ np.linalg.inv(self.eigvecs + 1e-10) @ x_prev
        xt = x_prev - step

        return xt
    
    def iterate_spreading_by_Julien(self):
        diffusion = [self.diffusion_init]  
        
        for _ in range(self.iterations):
            try:
                next_step = self.integration_step_by_Julien(diffusion[-1], self.timestep)
            except Exception as e:
                logging.error(e)
                break
            diffusion.append(next_step)  
            
        return np.asarray(diffusion) 
 
    def calc_exponent(self):
        ''' Inverse operation to log1p. '''
        self.cm = np.expm1(self.cm)
 
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

def run_simulation(subject, paths, output_dir, beta, delta_t, queue=None):    
    ''' Run simulation for single patient. '''

    subject_output_dir = os.path.join(output_dir, subject)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        return

    try:
        simulation = DiffusionSimulation(connect_matrix, beta, delta_t, t0_concentration)
        t1_concentration_pred = simulation.run()
        simulation.save_diffusion_matrix(subject_output_dir)
        simulation.save_terminal_concentration(subject_output_dir)
        '''
        visualize_diffusion_timeplot(simulation.diffusion_final.T, 
                                    simulation.timestep,
                                    simulation.t_total,
                                    save_dir=subject_output_dir)
        '''
        rmse = calc_rmse(t1_concentration_pred, t1_concentration)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]

    except Exception as e:
        logging.error(e)
        return
    
    '''
    visualize_terminal_state_comparison(t0_concentration, 
                                        min_t1_concentration_pred,
                                        t1_concentration,
                                        subject,
                                        min_rmse,
                                        opt_pcc,
                                        save_dir=subject_output_dir)
    '''
    if queue:
        queue.put([rmse, corr_coef])
    
### MULTIPROCESSING ###

if __name__ == '__main__':
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.error(e)
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path = f'../dataset_preparing/dataset_{category}.json'
    output_dir = '../../results'

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg RMSE", "SD RMSE", "Avg Pearson Correlation", "SD Pearson Correlation"]

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            num_cores = multiprocessing.cpu_count()
            logging.info(f"{num_cores} cores available")

    train_size = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    while train_size <= 0 or train_size > len(dataset.keys()):
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())}]: '))
        except Exception as e:
            logging.error(e)

    delta_t = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while delta_t <= 0:
        try:
            delta_t = int(input('Insert the time step you want to use [default is 0.01 seconds, equivalent to 7 days in a time window of 2 years]: '))
        except Exception as e:
            delta_t = 0.01
            logging.info(f"Using a timestep of {delta_t} seconds")

    N_fold = int(sys.argv[6]) if len(sys.argv) > 6 else -1
    while N_fold < 1:
        try:
            N_fold = int(input('Folds for cross validation: '))
        except Exception as e:
            logging.error(e)
            continue
    
    train_beta = []
    train_rmse = []
    train_pcc = []
    test_rmse = []
    test_pcc = []
    train_time = 0
    for i in tqdm(range(N_fold)):
        train_set = {}
        while len(train_set.keys()) < train_size:
            t = random.randint(0, len(dataset.keys())-1)
            if list(dataset.keys())[t] not in train_set.keys():
                train_set[list(dataset.keys())[t]] = dataset[list(dataset.keys())[t]]

        test_set = {}
        for subj, paths in dataset.items():
            if subj not in train_set:
                test_set[subj] = paths

        # Training ('beta' is calculated for each training patient and then averaged to be applied on testing set)
        procs = []
        start_time = time()
        queue = multiprocessing.Queue()
        beta_list = []
        for subj, paths in train_set.items():
            #logging.info(f"Patient {subj}")
            connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            #beta = (t1_concentration - t0_concentration) @ np.sum(connect_matrix, axis=0) * delta_t
            #beta_list.append(beta)
            '''
            p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, 1, 1, beta_iter, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
            
        for p in procs:
            p.join()
        '''
        
        #avg_beta = np.mean(beta_list)
    
        # Testing (use the learned 'avg_beta')
        procs = []
        queue = multiprocessing.Queue()
        for subj, paths in test_set.items():
            connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            avg_beta = (t1_concentration - t0_concentration) @ np.sum(connect_matrix, axis=0) * delta_t
            p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, avg_beta, delta_t, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
            
        for p in procs:
            p.join()
    
        # [opt_beta, min_rmse]
        while not queue.empty():
            err, pcc = queue.get()
            test_rmse.append(err)
            test_pcc.append(pcc)

        pt_avg.add_row([format(np.mean(test_rmse, axis=0), '.2f'), format(np.std(test_rmse, axis=0), '.2f'), format(np.mean(test_pcc, axis=0), '.2f'), format(np.std(test_pcc, axis=0), '.2f')])

        filename = f"../../results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_NDM_{category}.txt"
        out_file= open(filename, 'w')
        out_file.write(f"Category: {category}\n")
        out_file.write(f"Cores: {num_cores}\n")
        out_file.write(f"Subjects: {len(dataset.keys())}\n")
        out_file.write(f"Training set size: {train_size}\n")
        out_file.write(f"Testing set size: {len(dataset.keys()) - train_size}\n")
        out_file.write(f"Time step used (s): {delta_t}\n")
        out_file.write(f"Average beta used for testing: {avg_beta}\n")
        out_file.write(f"Folds: {N_fold}\n")
        out_file.write(f"Elapsed time for training (s): {format(train_time, '.2f')}\n")
        out_file.close()
        logging.info(f"Results saved in {filename}")