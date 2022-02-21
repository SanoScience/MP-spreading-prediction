''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

from audioop import rms
from collections import defaultdict
import os
from glob import glob
import logging
import random
from re import S
import sys
from time import time
import json
from matplotlib.pyplot import connect
from tqdm import tqdm 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as pearson_corr_coef
from utils_vis import *
from utils import *
from datetime import datetime
import multiprocessing
from prettytable import PrettyTable

date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
np.seterr(all = 'raise')

class MARsimulation:
    def __init__(self, connect_matrix, t0_concentrations, t1_concentrations, lam, iter_max=int(2e6)):
        self.N_regions = 166                                                    # no. of brain areas from the atlas
        self.iter_max = iter_max
        self.error_th = 0.01                                                    # acceptable error threshold for the reconstruction error
        self.gradient_th = 0.1                                                  # gradient difference threshold in stopping criteria in GD
        self.eta = 1e-10                                                         # learning rate of the gradient descent       
        self.cm = connect_matrix                                                # connectivity matrix 
        self.min_tract_num = 2                                                  # min no. of fibers to be kept (only when inverse_log==True)
        self.lam = lam
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations

    def run(self):
        ''' 
        Run simulation. 
        '''
        
        self.generate_indicator_matrix()
        pred_concentrations = None
        try:
            self.coef_matrix = self.run_gradient_descent() # get the model params
            pred_concentrations = self.coef_matrix @ self.init_concentrations # make predictions 
        except Exception as e:
            logging.error(e)

        return pred_concentrations
        
            
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
                gradient = -(self.final_concentrations - (A * self.B) @ self.init_concentrations) @ (self.init_concentrations.T * self.B) + self.lam * np.sum(np.abs(A)) 
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
                #self.eta+=1e-12
                prev_A = np.copy(A)
            except FloatingPointError:   
                self.eta = 1e-10
                A = np.copy(prev_A)
                logging.warning(f'Overflow encountered at iteration {iter_count}. Resetting learning rate to: {self.eta}')
                continue
                                          
        if vis_error: visualize_error(error_buffer)

        #logging.info(f"Final reconstruction error: {error_reconstruct}")
        #logging.info(f"Iterations: {iter_count}")
        return A
                  
def run_simulation(subject, paths, output_subj, connect_matrix, lam, iter_max, results_stem, queue):    
    ''' Run simulation for single patient. '''
      
    subject_output_subj = os.path.join(output_subj, subject)
    if not os.path.exists(subject_output_subj):
        os.makedirs(subject_output_subj)
    
    try:
        # load connectome ('is' works also with objects, '==' doesn't)
        if connect_matrix is None:
            connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome']))
            connect_matrix = prepare_cm(connect_matrix)
        
        # load proteins concentration in brain regions
        t0_concentration = load_matrix(paths['baseline']) 
        t1_concentration = load_matrix(paths['followup'])
        #logging.info(f'{subject} sum of t0 concentration: {np.sum(t0_concentration):.2f}')
        #logging.info(f'{subject} sum of t1 concentration: {np.sum(t1_concentration):.2f}')
    except Exception as e:
        logging.error(e)
        logging.error(f"Exception causing abortion of simulation for subject {subject}")

    rmse = corr_coef = None
    try:
        simulation = MARsimulation(connect_matrix, t0_concentration, t1_concentration, lam, iter_max)
        t1_concentration_pred = drop_negative_predictions(simulation.run())
        rmse = calc_rmse(t1_concentration, t1_concentration_pred)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(rmse) or np.isinf(rmse): raise Exception("Invalid value of RMSE")
        if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(f"Exception happened for \'simulation\' method of subject {subject}. Traceback:\n{e}") 
        
    save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subj, os.path.join(subject_output_subj, results_stem+'_prediction.png'), rmse, corr_coef)
    save_coeff_matrix(os.path.join(subject_output_subj,'A_'+results_stem+'.csv'), simulation.coef_matrix)

    queue.put([subject, rmse, corr_coef])

    return simulation.coef_matrix
           
### MULTIPROCESSING ###        

def parallel_training(dataset, output_subj, num_cores, lam, iter_max, dicts_subj):
    ''' 1st approach: train A matrix for each subject separately.
    The final matrix is an average matrix. '''
    file_stem = 'par_MAR'
    procs = []
    queue = multiprocessing.Queue()
    for subj, paths in tqdm(dataset.items()):
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, output_subj, None, lam, iter_max, file_stem, queue))
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
        subj, rmse, pcc = queue.get()
        dicts_subj[subj].append([rmse, pcc])
    
    conn_matrices = []
    # read results saved by "run simulation method"
    for subj, _ in dataset.items():
        conn_matrices.append(load_matrix(os.path.join(output_subj, subj, f'A_{file_stem}.csv')))
    
    avg_conn_matrix = np.mean(conn_matrices, axis=0)
    return avg_conn_matrix

def sequential_training(dataset, output_subj, lam, iter_max, dicts_subj):
    ''' 2nd approach: train A matrix for each subject sequentially (use the optimized matrix for the next subject)'''
    file_stem = 'seq_MAR'
    connect_matrix = None
    queue = multiprocessing.Queue()
    for subj, paths in tqdm(dataset.items()):
        tmp = None
        tmp = run_simulation(subj, paths, output_subj, connect_matrix, lam, iter_max, file_stem, queue)
        connect_matrix = tmp if tmp is not None else connect_matrix
    
    while not queue.empty():
        subj, rmse, pcc = queue.get()
        dicts_subj[subj].append([rmse, pcc])
    
    return connect_matrix

def test(conn_matrix, test_set, dicts_subj):
    rmse_list = []
    pcc_list = []
    for subj, paths in tqdm(test_set.items()):
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            pred = conn_matrix @ t0_concentration
            rmse = calc_rmse(t1_concentration, pred)
            pcc = pearson_corr_coef(t1_concentration, pred)[0]
            if np.isnan(rmse) or np.isinf(rmse): raise Exception("Invalid value of RMSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
        except Exception as e:
            logging.error(e)
            continue
        else:
            rmse_list.append(rmse)
            pcc_list.append(pcc)
            dicts_subj[subj].append([rmse, pcc])
    
    avg_rmse = np.mean(rmse_list, axis=0)
    avg_pcc = np.mean(pcc_list, axis=0)
    #logging.info(f"Average error on test samples for this fold: {avg_rmse}")
    #logging.info(f"Average Pearson correlation on test samples for this fold: {avg_pcc}")

    return avg_rmse, avg_pcc

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
    pt_avg.field_names = ["Type", "Avg RMSE", "SD RMSE", "Avg Pearson", "SD Pearson"]
    
    # Dictionary storing, for each patient (key), a list of couples (RMSE, PCC)
    dicts_subj_par = defaultdict(list)
    dicts_subj_seq = defaultdict(list)
    pt_subs = PrettyTable()
    pt_subs.field_names = ["Type", "ID", "Avg RMSE", "SD RMSE", "Avg Pearson", "SD Pearson"]
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

    train_size = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    while train_size <= 0 or train_size > len(dataset.keys()):
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())}]: '))
        except Exception as e:
            logging.error(e)
            
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while lam < 0 or lam > 1:
        try:
            lam = float(input('Insert the lambda coefficient for L1 penalty [0..1]: '))
        except Exception as e:
            logging.error(e)

    iter_max = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 10\'000]: '))
        except Exception as e:
            iter_max = 10000

    N_fold = int(sys.argv[6]) if len(sys.argv) > 6 else -1
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
        par_conn_matrix = parallel_training(train_set, output_subj, num_cores, lam, iter_max, dicts_subj_par)
        par_time += time() - start_time

        start_time = time()  
        seq_conn_matrix = sequential_training(train_set, output_subj, lam, iter_max, dicts_subj_seq)
        seq_time += time() - start_time

        rmse_par, pcc_par = test(par_conn_matrix, test_set, dicts_subj_par)
        total_rmse_par.append(rmse_par)
        total_pcc_par.append(pcc_par)

        rmse_seq, pcc_seq = test(seq_conn_matrix, test_set, dicts_subj_seq)
        total_rmse_seq.append(rmse_seq)
        total_pcc_seq.append(pcc_seq)

    # Note, these are the matrices from the last training phase (not the 'best of best')
    np.savetxt("results/A_matrix_par", par_conn_matrix, delimiter=',')
    np.savetxt("results/A_matrix_seq", seq_conn_matrix, delimiter=',')

    pt_avg.add_row(["Parallel", round(np.mean(total_rmse_par), 2), round(np.std(total_rmse_par), 2), round(np.mean(total_pcc_par), 2), round(np.std(total_pcc_par), 2)])
    pt_avg.add_row(["Sequential", round(np.mean(total_rmse_seq), 2), round(np.std(total_rmse_seq), 2), round(np.mean(total_pcc_seq), 2), round(np.std(total_pcc_seq), 2)])

    for subj in dicts_subj_par.keys():
        rmse_list = [el[0] for el in dicts_subj_par[subj]]
        pcc_list = [el[1] for el in dicts_subj_par[subj]]
        pt_subs.add_row(['PAR', subj, round(np.mean(rmse_list),2), round(np.std(rmse_list),2), round(np.mean(pcc_list),2), round(np.std(pcc_list),2)])
        
    for subj in dicts_subj_seq.keys():
        rmse_list = [el[0] for el in dicts_subj_seq[subj]]
        pcc_list = [el[1] for el in dicts_subj_seq[subj]]
        pt_subs.add_row(['SEQ', subj, round(np.mean(rmse_list),2), round(np.std(rmse_list),2), round(np.mean(pcc_list),2), round(np.std(pcc_list),2)])
    

    total_time = time() - total_time
    filename = f"{output_res}/{date}_MAR_{category}_{train_size}_{lam}_{iter_max}_{N_fold}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Lambda coefficient: {lam}\n")
    out_file.write(f"Iterations per patient: {iter_max}\n")
    out_file.write(f"Folds: {N_fold}\n")
    out_file.write(f"Elapsed time for \'Parallel\' training (s): {format(par_time, '.2f')}\n")
    out_file.write(f"Elapsed time for \'Sequential\' training (s): {format(seq_time, '.2f')}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string() + '\n')
    out_file.write(pt_subs.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")