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
from time import time
from turtle import shape
import pandas as pd

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian
from scipy.stats.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_log_error
from simulations.simulation_MAR import N_fold

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import load_matrix, calc_rmse, calc_msle
from datetime import datetime

import networkx as nx

logging.basicConfig(filename=f"../../results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_HKD_performance.txt", filemode='w', format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)

class DiffusionSimulation:
    def __init__(self, connect_matrix, beta, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.beta = beta 
        self.rois = 166 
        self.t_total = 2 # total length of the simulation in years
        self.timestep = 0.01 # equivalent to 7.3 days per time step
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
            logging.error(e)
        
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
        
    def integration_step(self, x0, t):
        # persistent mode of propagation
        # x(t) = U exp(-lambda * beta * t) U_conjugate x(0)
        # warning: t - elapsed time 
        # x0 is the initial configuration of the disease (baseline)
        #xt = self.eigvecs @ np.diag(np.exp(-self.eigvals * self.beta * t)) @ np.conjugate(self.eigvecs.T) @ x0    
           
        step = 1/(self.beta * self.eigvals +1e-5) * (1 - np.exp(-self.beta * self.eigvals * t)) * np.linalg.inv(self.eigvecs + 1e-5) * x0 + self.eigvecs
        xt = x0 + np.sum(step, axis=0) 
        return xt
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for i in range(self.iterations):
            next_step = self.integration_step(diffusion[i], self.timestep)
            diffusion.append(next_step)  
            
        return np.asarray(diffusion)   
    
    def integration_step_by_Julien(self, x_prev, timestep):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        # x(t)/dt = -B * H * x(t)
        #xt = x_prev - timestep * self.beta * self.L @ x_prev
        
        # where x(t) = e^(-B*H*t) * x0
        #xt = x_prev - timestep * self.beta * self.L @ x_prev

        # where x(t) = U * e^(-lambda*B*t) * U^(-1) * x0
        step = self.eigvecs * np.exp(-self.eigvals*self.beta*timestep) * np.linalg.inv(self.eigvecs + 1e-10) @ x_prev
        xt = x_prev - step

        return xt
    
    def iterate_spreading_by_Julien(self):
        diffusion = [self.diffusion_init]  
        
        for i in range(self.iterations):
            next_step = self.integration_step_by_Julien(diffusion[-1], self.timestep)
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

def run_simulation(subject, paths, output_dir, beta=1, step=1, N_runs=100, queue=None):    
    ''' Run simulation for single patient. '''

    subject_output_dir = os.path.join(output_dir, subject)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
      
    try:
        connect_matrix = load_matrix(paths['connectome'])
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
        logging.info(f'{subject} sum of t0 concentration: {np.sum(t0_concentration):.2f}')
        logging.info(f'{subject} sum of t1 concentration: {np.sum(t1_concentration):.2f}')
    except Exception as e:
        logging.error(e)
        return

    min_msle = -1
    opt_beta = None
    opt_pcc = None
    min_t1_concentration_pred = None
    for _ in range(N_runs):
        simulation = DiffusionSimulation(connect_matrix, beta, t0_concentration)
        try:
            t1_concentration_pred = simulation.run()
            simulation.save_diffusion_matrix(subject_output_dir)
            simulation.save_terminal_concentration(subject_output_dir)
        except Exception as e:
            logging.error(e)
            continue
        '''
        visualize_diffusion_timeplot(simulation.diffusion_final.T, 
                                    simulation.timestep,
                                    simulation.t_total,
                                    save_dir=subject_output_dir)
        '''
        msle = mean_squared_log_error(t1_concentration_pred, t1_concentration)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if msle < min_msle or min_msle == -1:
            min_msle = msle 
            opt_pcc = corr_coef
            min_t1_concentration_pred = t1_concentration_pred
            opt_beta = beta

        beta += step

    logging.info(f'Optimal beta was {opt_beta}')
    logging.info(f'Minimum MSLE for subject {subject} is: {min_msle:.2f}')
    logging.info(f'Corresponding Pearson correlation coefficient for subject {subject} is: {opt_pcc:.2f}')
    
    visualize_terminal_state_comparison(t0_concentration, 
                                        min_t1_concentration_pred,
                                        t1_concentration,
                                        subject,
                                        min_msle,
                                        opt_pcc,
                                        save_dir=subject_output_dir)
    if queue:
        queue.put([subject, opt_beta, min_msle])
    
def main():
    dataset_path = '../dataset_preparing/dataset_av45.json'
    output_dir = '../../results' 

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    num_cores = ''
    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
    except Exception as e:
        num_cores = multiprocessing.cpu_count()
    logging.info(f"{num_cores} cores available")

    N_runs = ''
    while not isinstance(N_runs, int) or N_runs < 0:
        try:
            N_runs = int(input('Number of iterations for Beta optimization: '))
        except Exception as e:
            logging.error(e)
            continue
    logging.info(f'Doing {N_runs} Beta optimization steps')

    train_size = -1
    while train_size <= 0 or train_size > len(dataset.keys()):
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())}]: '))
        except Exception as e:
            logging.error(e)
            continue
    logging.info(f"Train set of {train_size} elements")

    N_fold = ''
    while not isinstance(N_fold, int) or N_fold < 0:
        try:
            N_runs = int(input('Number of folds for cross validation of results: '))
        except Exception as e:
            logging.error(e)
            continue
    logging.info(f'Using {N_fold}-fold cross validation steps')
    
    train_beta = []
    train_msle = []
    test_msle = []
    for i in tqdm(range(N_fold)):
        logging.info(f"Fold {i+1}/{N_fold}")

        train_set = {}
        while len(train_set.keys()) < train_size:
            t = random.randint(0, len(dataset.keys())-1)
            if list(dataset.keys())[t] not in train_set.keys():
                train_set[list(dataset.keys())[t]] = dataset[list(dataset.keys())[t]]

        test_set = {}
        for subj, paths in dataset.items():
            if subj not in train_set:
                test_set[subj] = paths
        logging.info(f"Test set of {len(test_set)} elements")

        # Training ('beta' in increased of 'step' for 'N_runs' iterations)
        procs = []
        start_time = time()
        queue = multiprocessing.Queue()
        for subj, paths in train_set.items():
            logging.info(f"Patient {subj}")
            p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, 1, 1, N_runs, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
            
        for p in procs:
            p.join()

        train_time = time() - start_time
        logging.info(f"Training for {i}-th Fold done in {train_time} seconds")  
    
        # [subject, opt_beta, min_msle]
        for subj, b, err in queue:
            train_beta.append(b)
            train_msle.append(err)

        avg_beta = np.mean(train_beta, axis=0)
        train_avg_msle = np.mean(train_msle, axis=0)

        logging.info(f"Average Beta from training set: {avg_beta}")
        logging.info(f"MSLE values on training set:\n{train_msle}")
        logging.info(f"Average MSLE on training set: {train_avg_msle}")
    
        # Testing (use the learned 'avg_beta' without changing it)
        procs = []
        start_time = time()
        queue = multiprocessing.Queue()
        for subj, paths in test_set.items():
            logging.info(f"Patient {subj}")
            p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, avg_beta, 0, 1, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
            
        for p in procs:
            p.join()

        test_time = time() - start_time
        logging.info(f"Testing for {i}-th Fold done in {test_time} seconds")  
    
        # [opt_beta, min_msle]
        for subj, _, err in queue:
            test_msle.append(err)
        test_avg_msle = np.mean(test_msle, axis=0)
        logging.info(f"Average MSLE on test set (with beta={avg_beta}): {test_avg_msle}")
        logging.info(f"MSLE values on test set:\n{test_msle}")
    
if __name__ == '__main__':
    main()