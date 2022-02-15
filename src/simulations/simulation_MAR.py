''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

from audioop import rms
import os
from glob import glob
import logging
import random
import sys
from time import time
import json
from black import out
from tqdm import tqdm 
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr as pearson_corr_coef
from utils_vis import *
from utils import *
from datetime import datetime
import multiprocessing
from prettytable import PrettyTable

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
np.seterr(all = 'raise')

class MARsimulation:
    def __init__(self, connect_matrix, t0_concentrations, t1_concentrations, iter_max=int(2e6)):
        self.N_regions = 166                                                    # no. of brain areas from the atlas
        self.iter_max = iter_max
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

        while (error_reconstruct > self.error_th) and iter_count < self.iter_max:
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
                  
def run_simulation(subject, paths, output_dir, connect_matrix, make_plot, save_results, iter_max, q = None):    
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
        simulation = MARsimulation(connect_matrix, t0_concentration, t1_concentration, iter_max)
        t1_concentration_pred = drop_negative_predictions(simulation.run(norm_opt=2))
        error = calc_rmse(t1_concentration, t1_concentration_pred)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if make_plot: visualize_terminal_state_comparison(  t0_concentration, 
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
           
### MULTIPROCESSING ###        

def parallel_training(dataset, output_dir, num_cores, iter_max):
    ''' 1st approach: train A matrix for each subject separately.
    The final matrix is an average matrix. '''
    procs = []
    for subj, paths in dataset.items():
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_dir, None, False, True, iter_max))
        p.start()
        procs.append(p)        
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                p.join(timeout=10)
                if not p.is_alive():
                    procs.remove(p)         
        for p in procs:
            p.join()
    
    conn_matrices = []
    # read results saved by "run simulation method"
    for subj, _ in dataset.items():
        conn_matrices.append(load_matrix(os.path.join(output_dir, subj, 'A_matrix_MAR.csv')))
    
    avg_conn_matrix = np.mean(conn_matrices, axis=0)
    return avg_conn_matrix

def sequential_training(dataset, output_dir, iter_max):
    ''' 2nd approach: train A matrix for each subject sequentially (use the optimized matrix for the next subject)'''
    connect_matrix = None
    for subj, paths in dataset.items():
        tmp = None
        tmp = run_simulation(subj, paths, output_dir, connect_matrix, False, True, iter_max)
        connect_matrix = tmp if tmp is not None else connect_matrix
    
    return connect_matrix

def test(conn_matrix, test_set):
    rmse_list = []
    pcc_list = []
    for subj, paths in test_set.items():
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            pred = conn_matrix @ t0_concentration
            rmse = calc_rmse(t1_concentration, pred)
            pcc = pearson_corr_coef(t1_concentration, pred)[0]
        except Exception as e:
            logging.error(e)
            logging.error(f"Error in loading data from patient {subj}, skipping...")
        else:
            rmse_list.append(rmse)
            pcc_list.append(pcc)
    
    avg_rmse = np.mean(rmse_list, axis=0)
    avg_pcc = np.mean(pcc_list, axis=0)
    logging.info(f"Average error on test samples for this fold: {avg_rmse}")
    logging.info(f"Average Pearson correlation on test samples for this fold: {avg_pcc}")

    return avg_rmse, avg_pcc

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
    pt_avg.field_names = ["Type", "Avg RMSE", "SD RMSE", "Avg Pearson", "SD Pearson"]

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = sys.argv[2] if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            num_cores = multiprocessing.cpu_count()
            logging.info(f"{num_cores} cores available")

    train_size = sys.argv[3] if len(sys.argv) > 3 else -1
    while train_size <= 0 or train_size > len(dataset.keys()):
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())}]: '))
        except Exception as e:
            logging.error(e)

    iter_max = sys.argv[4] if len(sys.argv) > 4 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 10\'000]: '))
        except Exception as e:
            iter_max = 10000

    N_fold = sys.argv[5] if len(sys.argv) > 5 else -1
    while N_fold < 1:
        try:
            N_fold = int(input('Folds for cross validation: '))
        except Exception as e:
            logging.error(e)
            continue

    total_rmse_par = []
    total_pcc_par = []

    total_rmse_seq = []
    total_pcc_seq = []

    par_time = 0
    seq_time = 0
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

        start_time = time()
        par_conn_matrix = parallel_training(train_set, output_dir, num_cores, iter_max)
        par_time += time() - start_time

        start_time = time()  
        seq_conn_matrix = sequential_training(train_set, output_dir, iter_max)
        seq_time += time() - start_time

        rmse_par, pcc_par = test(par_conn_matrix, test_set)
        total_rmse_par.append(rmse_par)
        total_pcc_par.append(pcc_par)

        rmse_seq, pcc_seq = test(seq_conn_matrix, test_set)
        total_rmse_seq.append(rmse_seq)
        total_pcc_seq.append(pcc_seq)
    
    avg_rmse_par = np.mean(total_rmse_par, axis=0)
    avg_pcc_par = np.mean(total_pcc_par, axis=0)

    avg_rmse_seq = np.mean(total_rmse_seq, axis=0)
    avg_pcc_seq = np.mean(total_pcc_seq, axis=0)

    # Note, these are the matrices from the last training phase (not the 'best of best')
    np.savetxt("../../results/A_matrix_par", par_conn_matrix, delimiter=',')
    np.savetxt("../../results/A_matrix_seq", seq_conn_matrix, delimiter=',')

    pt_avg.add_row(["Parallel", format(avg_rmse_par, '.2f'), "", format(avg_pcc_par, '.2f'), ""])
    pt_avg.add_row(["Sequential", format(avg_rmse_seq, '.2f'), "", format(avg_pcc_seq, '.2f'), ""])

    filename = f"../../results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_MAR_{category}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Iterations per patient: {iter_max}\n")
    out_file.write(f"Folds: {N_fold}\n")
    out_file.write(f"Elapsed time for \'Parallel\' training (s): {format(par_time, '.2f')}\n")
    out_file.write(f"Elapsed time for \'Sequential\' training (s): {format(seq_time, '.2f')}\n")
    out_file.write(pt_avg.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")