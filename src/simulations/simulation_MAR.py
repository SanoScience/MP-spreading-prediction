''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

import os
from glob import glob
import logging
import random
from time import time
import warnings
import concurrent.futures
import json

from tqdm import tqdm 
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import *
from utils import *
from datetime import datetime

import multiprocessing

logging.basicConfig(filename=f"../../results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_MAR_performance.txt", filemode='w', format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
np.seterr(all = 'raise')

class MARsimulation:
    def __init__(self, connect_matrix, t0_concentrations, t1_concentrations, maxiter=int(2e6)):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.N_regions = 166                                                    # no. of brain areas from the atlas
        self.maxiter = maxiter
        self.error_th = 0.01                                                    # acceptable error threshold for the reconstruction error
        self.gradient_th = 0.1                                                  # gradient difference threshold in stopping criteria in GD
        self.eta = 1e-10                                                         # learning rate of the gradient descent       
        self.cm = connect_matrix                                                # connectivity matrix 
        self.min_tract_num = 2                                                  # min no. of fibers to be kept (only when inverse_log==True)
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations

    def run(self, norm_opt, inverse_log=True):
        ''' 
        Run simulation. 
        Args:
            norm_opt (int): normalize option for connectivity matrix
            inverse_log (boolean): if True use normal values instead of logarithmic in connectivity matrix '''
        if inverse_log: 
            try:
                self.calc_exponent()
                self.filter_connections()
            except Exception as e:
                logging.error(e)
        
        self.transform_cm(norm_opt)
        self.generate_indicator_matrix()
        pred_concentrations = None
        try:
            self.coef_matrix = self.run_gradient_descent() # get the model params
            pred_concentrations = self.coef_matrix @ self.init_concentrations # make predictions 
        except Exception as e:
            logging.error(e)

        return pred_concentrations
 
    def calc_exponent(self):
        ''' 
        Transform the connectivity matrix to get the no. of connections between regions
        instead of logarithm. 
        
        Inverse operation to log1p. '''
        try:
            self.cm = np.expm1(self.cm)
        except FloatingPointError as e:
            logging.error(e)
            logging.error("Overflow encountered, using the inaltered matrix...")
        
    def filter_connections(self):
        ''' Filter out all connections with less than n fiber reaching. '''
        self.cm = np.where(self.cm > self.min_tract_num, self.cm, 0).astype('float32')
        
    def transform_cm(self, norm_opt):
        '''
        Transform initial conenctivity matrix. 
        
        Args:
            0 means no normalization, = 1 means binarize, = 2 means divide by the maximum '''
        
        if norm_opt == 0:
            #logging.info('No normalization of the initial matrix')
            pass
        elif norm_opt == 1:
            #logging.info('Initial matrix binarized')
            self.cm = np.where(self.cm > 0, 1, 0).astype('float32')
        elif norm_opt == 2:
            #logging.info('Initial matrix normalized according to its largest value')
            max_val = np.max(self.cm)
            self.cm /= max_val
        else:
            pass
            #logging.info('No normalization of the initial matrix')
            
    def generate_indicator_matrix(self):
        ''' Construct a matrix with only zeros and ones to be used to 
        reinforce the zero connection (this is **B** in our paper).
        B has zero elements where no structural connectivity appears. '''
        self.B = np.where(self.cm==0, 0, 1).astype('float32')
        
    def run_gradient_descent(self, vis_error=False):
        iter_count = 0                                                          # counter of the current iteration 
        error_reconstruct = 1e10                                                # initial error of reconstruction gradients)
        if vis_error: error_buffer = []                                         # reconstruction error along iterations
        A = self.cm                                                             # the resulting effective matrix; initialized with connectivity matrix; [N_regions x N_regions]
        gradient = np.ones((self.N_regions, self.N_regions)) 
        prev_A = np.copy(A)

        while (error_reconstruct > self.error_th) and iter_count < self.maxiter:
            try:
                # calculate reconstruction error 
                error_reconstruct = 0.5 * np.linalg.norm(self.final_concentrations - (A * self.B) @ self.init_concentrations, ord=2)**2
                if vis_error: error_buffer.append(error_reconstruct)
                
                # gradient computation
                gradient = -(self.final_concentrations - (A * self.B) @ self.init_concentrations) @ (self.init_concentrations.T * self.B) 
                A -= self.eta * gradient       
                # reinforce where there was no connection at the beginning 
                A *= self.B

                '''
                norm = np.linalg.norm(gradient)
                        
                if norm < self.gradient_th:
                    logging.info(f"Gradient norm: {norm}.\nTermination criterion met, quitting...")
                    break    

                if iter_count % 100000 == 0:
                    logging.info(f'Gradient norm at {iter_count}th iteration: {norm:.2f} (current eta {self.eta})')
                '''

                iter_count += 1
                self.eta+=1e-12
                prev_A = np.copy(A)
                
            except FloatingPointError:   
                self.eta = 1e-10
                A = np.copy(prev_A)
                logging.warning(f'Overflow encountered at iteration {iter_count}. Changing starting learning rate to: {self.eta}')
                continue
                                          
        if vis_error: visualize_error(error_buffer)

        logging.info(f"Final reconstruction error: {error_reconstruct}")
        #logging.info(f"Iterations: {iter_count}")
        return A
                   
def run_simulation(subject, paths, output_dir, connect_matrix, make_plot, save_results, maxiter):    
    ''' Run simulation for single patient. '''
      
    subject_output_dir = os.path.join(output_dir, subject)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
    
    try:
        # load connectome ('is' works also with objects, '==' doesn't)
        if connect_matrix is None:
            connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
        
        # load proteins concentration in brain regions
        t0_concentration = load_matrix(paths['baseline']) 
        t1_concentration = load_matrix(paths['followup'])
        logging.info(f'{subject} sum of t0 concentration: {np.sum(t0_concentration):.2f}')
        logging.info(f'{subject} sum of t1 concentration: {np.sum(t1_concentration):.2f}')
    except Exception as e:
        logging.error(e)
        logging.error(f"Exception causing abortion of simulation for subject {subject}")

    try:
        simulation = MARsimulation(connect_matrix, t0_concentration, t1_concentration, maxiter)
        t1_concentration_pred = drop_negative_predictions(simulation.run(norm_opt=2))

        error = calc_rmse(t1_concentration, t1_concentration_pred)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if make_plot: visualize_terminal_state_comparison(t0_concentration, 
                                            t1_concentration_pred,
                                            t1_concentration,
                                            subject,
                                            error, 
                                            corr_coef)
        if save_results:
            save_terminal_concentration(subject_output_dir, t1_concentration_pred, 'MAR')
            save_coeff_matrix(subject_output_dir, simulation.coef_matrix)
    except Exception as e:
        logging.error(f"Exception happened for \'simulation\' method of subject {subject}. Traceback:\n{e}") 
        return None       

    return simulation.coef_matrix
           
def parallel_training(dataset, output_dir, num_cores, maxiter):
    ''' 1st approach: train A matrix for each subject separately.
    The final matrix is an average matrix. '''
    procs = []

    for subj, paths in dataset.items():
        #dispatcher(files[i], atlas_file, img_type)
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, None, False, True, maxiter))
        p.start()
        procs.append(p)
        
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                # wait for 10 seconds to wait process termination
                p.join(timeout=10)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
        
        # wait the last chunk            
        for p in procs:
            p.join()
    
    conn_matrices = []
    # read results saved by "run simulation method"
    for subj, _ in dataset.items():
        conn_matrices.append(load_matrix(os.path.join(output_dir, subj, 'A_matrix_MAR.csv')))
    
    avg_conn_matrix = np.mean(conn_matrices, axis=0)
    return avg_conn_matrix

def sequential_training(dataset, output_dir, maxiter):
    ''' 2nd approach: train A matrix for each subject sequentially (use the optimized matrix for the next subject)'''

    connect_matrix = None
    for subj, paths in dataset.items():
        tmp = run_simulation(subj, paths, output_dir, connect_matrix, False, True, maxiter)
        connect_matrix = tmp if tmp is not None else connect_matrix
    
    return connect_matrix

def test(conn_matrix, test_set):
    errors = pd.DataFrame(index=test_set.keys(), columns=['RMSE'])
    for subj, paths in test_set.items():
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            pred = conn_matrix @ t0_concentration
            errors.loc[subj] = calc_rmse(t1_concentration, pred)
        except Exception as e:
            logging.error(e)
            logging.error(f"Error in loading data from patient {subj}, skipping...")
    
    logging.info(errors)
    logging.info(f"Average error on test samples for this fold: {errors['RMSE'].mean()}")

    return errors

if __name__ == '__main__':
    # TODO: iterate for all tracers (or ask to the user)
    dataset_path = '../dataset_preparing/dataset_av45.json'
    output_dir = '../../results'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
    except Exception as e:
        num_cores = multiprocessing.cpu_count()
    logging.info(f"{num_cores} cores available")

    train_size = -1
    while train_size <= 0 or train_size > len(dataset.keys()):
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())}]: '))
        except Exception as e:
            logging.error(e)
            continue
    logging.info(f"Train set of {train_size} elements")

    try:
        maxiter = int(input(f'Number of maximum iterations for each simulation [default {int(2e6)}]: '))
    except Exception as e:
        logging.error(e)
        maxiter = int(2e6)

    logging.info(f"Maximum iterations for each simulation: {maxiter}")

    N_fold = ''
    while not isinstance(N_fold, int) or N_fold < 0:
        try:
            N_fold = int(input('Folds for cross validation: '))
        except Exception as e:
            logging.error(e)
            continue
    logging.info(f'Using {N_fold}-folds')

    performance_par = []
    performance_seq = []

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

        start_time = time()
        par_conn_matrix = parallel_training(train_set, output_dir, num_cores, maxiter)
        par_time = time() - start_time
        logging.info("Parallel Training for {i}-th Fold done in {par_time} seconds")  

        start_time = time()  
        seq_conn_matrix = sequential_training(train_set, output_dir, maxiter)
        seq_time = time() - start_time
        logging.info(f"Sequential Training for {i}-th Fold done in {par_time} seconds")

        performance_par.append(test(par_conn_matrix, test_set))
        performance_seq.append(test(seq_conn_matrix, test_set))
    
    pd_par_tot = pd.concat(performance_par)
    pd_seq_tot = pd.concat(performance_seq)

    logging.info("Mean RMSE on the whole dataset")
    logging.info(f"Parallel: {pd_par_tot['RMSE'].mean()}")
    logging.info(f"Sequencial: {pd_seq_tot['RMSE'].mean()}")

    #TODO: do it for distinct categories (AD, LMCI, EMCI, CN)