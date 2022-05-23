"""
    SYNOPSIS
    python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <use_binary> <iter_max> <N_fold>
"""

''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

from collections import defaultdict
import os
import logging
import random
import sys
from time import time
import json
from tqdm import tqdm 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as pearson_corr_coef
from utils_vis import *
from utils import *
from datetime import datetime
import multiprocessing
from prettytable import PrettyTable
import yaml
from sklearn.metrics import mean_squared_error

date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
np.seterr(all = 'raise')

class MARsimulation:
    def __init__(self, subject, connect_matrix, t0_concentrations, t1_concentrations, lam, user_binary, iter_max=int(2e6)):
        self.subject = subject
        self.cm = connect_matrix
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations
        self.lam = lam
        self.use_binary = user_binary
        self.iter_max = iter_max

        self.N_regions = 166            # no. of brain areas from the atlas
        self.error_th = 0            # acceptable error threshold for the reconstruction error
        self.gradient_th = 0         # gradient difference threshold in stopping criteria in GD
        self.eta = 1e-3                 # learning rate of the gradient descent       
        self.backup_eta = 1e-7                         
        self.eta_step = 1e-4
        self.max_retry = 10  

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
            logging.error(f"Exception during gradient descent for subject {self.subject}")

        return pred_concentrations
        
            
    def generate_indicator_matrix(self):
        ''' Construct a matrix with only zeros and ones to be used to 
        reinforce the zero connection (this is **B** in our paper).
        B has zero elements where no structural connectivity appears. '''
        # NOTE: B is a matrix of ones (EXPERIMENTAL)
        if self.use_binary:
            self.B = np.where(self.cm>0, 1, 0)
        else:
            self.B = np.ones_like(self.cm)

    def run_gradient_descent(self, vis_error=False):                           
        error_reconstruct = 1 + self.error_th                                 
        if vis_error: error_buffer = []                                         # reconstruction error along iterations
        trial = 0
        while trial<self.max_retry:
            iter_count = 0                               
            #A = np.copy(self.cm)
            A = np.ones_like(self.cm)
            while (error_reconstruct > self.error_th) and iter_count < self.iter_max:
                try:
                    # NOTE: numpy.linalg.norm has a parameter 'ord' which specifies the kind of norm to compute
                    # ord=1 corresponds to max(sum(abs(x), axis=0))
                    # ord=None corresponds to Frobenius norm (square root of the sum of the squared elements)

                    # calculate reconstruction error 
                    error_reconstruct = 0.5 * np.linalg.norm(self.final_concentrations - (A * self.B) @ self.init_concentrations)**2
                    if vis_error: error_buffer.append(error_reconstruct)
                except FloatingPointError as e:
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during computation of reconstruction error (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    # gradient computation
                    gradient = -(self.final_concentrations - A @ self.init_concentrations) @ self.init_concentrations.T + self.lam * np.linalg.norm(A)  
                    
                    
                    norm = np.linalg.norm(gradient)
                    if norm <= self.gradient_th:
                        logging.info(f"Gradient norm: {norm}.\nTermination criterion met for subject {self.subject}, quitting...")
                        break    
                    
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during gradient computation (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    A = (A - (self.eta * gradient)) * self.B
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during updating of coefficient matrix (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break
                
                self.eta += self.eta_step
                iter_count += 1
                
            if (error_reconstruct <= self.error_th) or iter_count == self.iter_max or norm <= self.gradient_th: break
            self.eta = self.backup_eta
            self.eta_step /= 10
            trial += 1
        
        if trial == self.max_retry: logging.error(f"Subject {self.subject} couldn't complete gradient descent")
                                          
        if vis_error: visualize_error(error_buffer)

        logging.info(f"Final reconstruction error for subject {self.subject}: {error_reconstruct} ({iter_count} iterations)")
        return A
                  
def run_simulation(subject, paths, connect_matrix, lam, use_binary, iter_max, queue):    
    ''' Run simulation for single patient. '''
    
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['CM']))
        #connect_matrix = prepare_cm(connect_matrix)

        t0_concentration = load_matrix(paths['baseline']) 
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        logging.error(f"Exception during loading of CM/ for subject {subject}")
        queue.put([subject, -1, -1])
        return

    mse = pcc = None
    try:
        simulation = MARsimulation(subject, connect_matrix, t0_concentration, t1_concentration, lam, use_binary, iter_max)
    except Exception as e:
        logging.error(f"Exception happened during initialization of 'MARsimulation' object for subject {subject}. Traceback:\n{e}") 
        queue.put([subject, -1, -1])
        return

    try:
        t1_concentration_pred = drop_negative_predictions(simulation.run())
    except Exception as e:
        logging.error(f"Exception happened for during execution of simulation for subject {subject}. Traceback:\n{e}") 
        queue.put([subject, -1, -1])
        return 

    try:
        mse = mean_squared_error(t1_concentration, t1_concentration_pred)
        pcc = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
        if np.isnan(pcc): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(f"Exception happened for Pearson correlation coefficient and MSE computation for subject {subject}. Traceback:\n{e}") 
        queue.put([subject, -1, -1])
        return
    
    try:
        save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subject, subject +'CMAR.png', mse, pcc)
        save_coeff_matrix(subject +'CMAR.csv', simulation.coef_matrix)
    except Exception as e:
        logging.error(f"Exception happened during saving of simulation results for subject {subject}. Traceback:\n{e}") 
        queue.put([subject, -1, -1])
        return

    queue.put([subject, mse, pcc])

    return
           
### MULTIPROCESSING ###        

def training(dataset, num_cores, lam, use_binary, iter_max, dicts_subj):
    ''' 1st approach: train A matrix for each subject separately.
    The final matrix is an average matrix. '''
    procs = []
    queue = multiprocessing.Queue()
    counter = 0
    done = 0
    for subj, paths in tqdm(dataset.items()):
        p = multiprocessing.Process(target=run_simulation, args=(subj, paths, None, lam, use_binary, iter_max, queue))
        p.start()
        procs.append(p)        
        counter += 1
        if counter%num_cores == 0 and counter > 0:
            subj, mse, pcc = queue.get()
            # mse and pcc are -1 if an exception happened during simulation, and are therefore not considered
            if mse != -1 and pcc != -1:
                dicts_subj[subj].append([mse, pcc])
            counter -= 1
            done += 1

    while done < len(procs):
        subj, mse, pcc = queue.get()
        # mse and pcc are -1 if an exception happened during simulation, and are therefore not considered
        if mse != -1 and pcc != -1:
            dicts_subj[subj].append([mse, pcc])
        done += 1
    
    conn_matrices = []
    # read results saved by "run simulation method"
    for subj, _ in dataset.items():
        try:
            conn_matrices.append(load_matrix(subj + 'CMAR.csv'))
        except Exception as e:
            logging.error(f"Matrix for subject {subj} could not be found")

    avg_conn_matrix = np.mean(conn_matrices, axis=0)
    return avg_conn_matrix

''' DEPRECATED (INEFFICIENT)
    2nd approach: train A matrix for each subject sequentially (use the optimized matrix for the next subject)
'''

def test(conn_matrix, test_set, dicts_subj):
    mse_list = []
    pcc_list = []
    for subj, paths in tqdm(test_set.items()):
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            pred = conn_matrix @ t0_concentration
            mse = mean_squared_error(t1_concentration, pred)
            pcc = pearson_corr_coef(t1_concentration, pred)[0]
            if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
            save_prediction_plot(t0_concentration, pred, t1_concentration, subj, subj +'CMAR.png', mse, pcc)
        except Exception as e:
            logging.error(e)
            continue
        else:
            mse_list.append(mse)
            pcc_list.append(pcc)
            dicts_subj[subj].append([mse, pcc])
    
    avg_mse = np.mean(mse_list, axis=0)
    avg_pcc = np.mean(pcc_list, axis=0)
    #logging.info(f"Average error on test samples for this fold: {avg_mse}")
    #logging.info(f"Average Pearson correlation on test samples for this fold: {avg_pcc}")

    return avg_mse, avg_pcc

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

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
    
    # Dictionary storing, for each patient (key), a list of couples (MSE, PCC)
    dicts_subj = defaultdict(list)
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson", "Set"]
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
    while train_size <= 0 or train_size > len(dataset.keys())-1:
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())-1}]: '))
        except Exception as e:
            logging.error(e)
            
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while lam < 0 or lam > 1:
        try:
            lam = float(input('Insert the lambda coefficient for L1 penalty [0..1]: '))
        except Exception as e:
            logging.error(e)

    use_binary = bool(sys.argv[5]) if len(sys.argv) > 5 else -1
    while use_binary < 0:
        try:
            use_binary = False if input('Choose if using binary matrix [1] or not [0] [default 1]: ') == '0' else True
        except Exception as e:
            use_binary = True

    iter_max = int(sys.argv[6]) if len(sys.argv) > 6 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 1\'000\'000]: '))
        except Exception as e:
            iter_max = 1e6

    N_fold = int(sys.argv[7]) if len(sys.argv) > 7 else -1
    while N_fold < 1:
        try:
            N_fold = int(input('Folds for cross validation: '))
        except Exception as e:
            logging.error(e)
            continue

    total_mse = []
    total_pcc = []

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

        start_time = time()
        coeff_matrix = training(train_set, num_cores, lam, use_binary, iter_max, dicts_subj)
        train_time += time() - start_time

        mse, pcc = test(coeff_matrix, test_set, dicts_subj)
        total_mse.append(mse)
        total_pcc.append(pcc)

    # Note, these are the matrices from the last training phase (not the 'best of best')
    np.savetxt(output_res+'MAR.csv', coeff_matrix, delimiter=',')

    pt_avg.add_row([round(np.mean(total_mse), 5), round(np.std(total_mse), 2), round(np.mean(total_pcc), 5), round(np.std(total_pcc), 2)])

    for subj in dicts_subj.keys():
        mse_list = [el[0] for el in dicts_subj[subj]]
        pcc_list = [el[1] for el in dicts_subj[subj]]
        sub_set = 'Training' if subj in train_set.keys() else 'Testing'
        pt_subs.add_row([subj, round(np.mean(mse_list), 5), round(np.std(mse_list), 2), round(np.mean(pcc_list), 5), round(np.std(pcc_list), 2), sub_set])

    total_time = time() - total_time
    filename = f"{output_res}{date}_MAR_{category}_{train_size}_{lam}_{iter_max}_{N_fold}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Lambda coefficient: {lam}\n")
    out_file.write(f"Binary matrix: {'yes' if use_binary else 'no'}\n")
    out_file.write(f"Iterations per patient: {iter_max}\n")
    out_file.write(f"Folds: {N_fold}\n")
    out_file.write(f"Elapsed time for training (s): {format(train_time, '.2f')}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string() + '\n')
    out_file.write(pt_subs.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")