"""
SYNOPSIS
python3 simulation_CDRMR.py <category> <cores> <train_size> <iter_max> <N_fold>


Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
"""

import os
import logging
import sys
from time import time
import json
from tqdm import tqdm 
import numpy as np
from utils_vis import *
from utils import *
from datetime import datetime
from threading import Thread, Lock
from multiprocessing import cpu_count
from prettytable import PrettyTable
import yaml

np.seterr(all = 'raise')
date = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_CDRMR_{date}.log")
digits = 5

class CDRMR(Thread):
    def __init__(self, subject, paths):
        Thread.__init__(self)
        self.subject = subject
        self.paths = paths

        self.error_stop = 1e-15                                                  # acceptable error threshold for the reconstruction error
        self.gradient_thr = 1e-10
        self.eta = 1e-3                                            
        self.max_retry = 10 
        self.max_bad_iter = 1     
    
    def generate_A(self):
        ''' Generate the matrix A from which starting the optimization (this is **A** in our paper). '''
        return np.ones_like(self.t0_concentration) + self.t0_cdr
    
    def gradient_descent(self):                           
        error_reconstruct = 1 + self.error_stop         
        previous_error = -1             
 
        trial = 0
        
        while trial<self.max_retry:
            iter_count = 0     
            bad_iter_count = 0                          
            self.A = self.generate_A()
            while (error_reconstruct > self.error_stop) and iter_count < iter_max:
                try:
                    # NOTE: numpy.linalg.norm has a parameter 'ord' which specifies the kind of norm to compute
                    # ord=1 corresponds to max(sum(abs(x), axis=0))
                    # ord=None corresponds to Frobenius norm (square root of the sum of the squared elements)

                    # calculate reconstruction error (NOTE conversely to MAR, the algebraic norm is not appliable)                 
                    error_reconstruct = np.abs(self.t1_cdr - self.A @ self.t0_concentration)
                    
                    if previous_error != -1 and previous_error <= error_reconstruct:
                        bad_iter_count += 1 
                        if bad_iter_count >= self.max_bad_iter:
                            # revert A to the previous step (if antigradient diection (-) caused an increment, follow the gradient (+)direction)
                            self.A = self.A + (self.eta * gradient)
                            logging.info("Error is raising, halting optimization")
                            break

                    previous_error = error_reconstruct
                   
                except FloatingPointError as e:
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during computation of reconstruction error (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    gradient = - (self.t1_cdr - self.A @ self.t0_concentration + self.t0_cdr)
                    norm = np.linalg.norm(gradient)
                    if norm <= self.gradient_thr:
                        logging.info(f"Gradient for subject {self.subject} met termination criterion")
                        break
                    
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during gradient computation (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    self.A = self.A - (self.eta * gradient)
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during updating of coefficient matrix (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break
                
                iter_count += 1
                
            if error_reconstruct <= self.error_stop or bad_iter_count == self.max_bad_iter or iter_count == iter_max or norm <= self.gradient_thr: break
            trial += 1
            self.eta /= 10 
        
        if trial == self.max_retry: logging.error(f"Subject {self.subject} couldn't complete gradient descent")           
        logging.info(f"Final reconstruction error for subject {self.subject}: {error_reconstruct} ({iter_count} iterations)")
    
        return 
                  
    def run(self):    
        ''' Run simulation for single patient. '''

        # LOAD DATA
        try:
            self.t0_concentration = load_matrix(self.paths['baseline']) 
            self.t0_cdr = float(self.paths['CDR_t0_score'])
            self.t1_cdr = float(self.paths['CDR_t1_score'])
        except Exception as e:
            logging.error(e)
            logging.error(f"Exception during loading of data for subject {self.subject}")
            return

        # TRAIN 
        try:
            self.gradient_descent()
        except Exception as e:
            logging.error(e)
            logging.error(f"Exception during gradient descent for subject {self.subject}")

        # COMPUTE TRAIN STATISTICS
        try:
            t1_cdr_pred = self.A @ self.t0_concentration # make prediction
            logging.info(f"Computiong PCC and ABSERR for subject {self.subject}")
            ae = abs(self.t1_cdr - t1_cdr_pred)
            # we can't compute the Pearson correlation coefficient for a single value
            #pcc = pearson_corr_coef(self.t1_cdr, t1_cdr_pred)[0]
            if np.isnan(ae) or np.isinf(ae): raise Exception("Invalid value of ABSERR")
            #if np.isnan(pcc): raise Exception("Invalid value of PCC")
            ae = round(ae, digits)
            #pcc = round(pcc, digits)
        except Exception as e:
            logging.error(f"Exception happened for ABSERR computation for subject {self.subject}. Traceback:\n{e}") 
            return

        lock.acquire()
        
        # NOTE: plots should be saved one by one without multiple threads!
        try:
            #logging.info(f"Saving prediction plot for subject {self.subject}")
            #save_prediction_plot(self.t0_concentration, t1_cdr_pred, self.t1_cdr, self.subject, self.subject + 'train/CDRMR_train_' + date + '.png', ae)
            save_coeff_matrix(self.subject + 'train/CDRMR_train_' + date + '.csv', self.A)
            A_matrices[self.subject] = self.A
        except Exception as e:
            logging.error(f"Exception happened during saving of simulation results for subject {self.subject}. Traceback:\n{e}")
            
        training_scores[self.subject] = ae
        
        lock.release()

        return   

def training():
    
    works = []
    for subj, paths in train_set.items():
        if subj in A_matrices: continue
        logging.info(f"Subject {subj} is not in A_matrices, queuing for training")
        works.append(CDRMR(subj, paths))
        works[-1].start()      

        while len(works) >= num_cores:
            for w in works:
                if not w.is_alive(): 
                    works.remove(w)
    
    for w in works:
        w.join()
    
    train_matrices = []
    # read results saved by "run_simulation method"
    for subj, _ in train_set.items():
        try:
            train_matrices.append(A_matrices[subj])
        except Exception as e:
            logging.error(f"Matrix for subject {subj} with a suitable date could not be found")

    return train_matrices

def test(train_matrices):
    ae_subj = []
    for subj, paths in test_set.items():
        logging.info(f"Test on subject {subj}")
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t0_cdr = float(paths['CDR_t0_score'])
            t1_cdr = float(paths['CDR_t1_score'])
            predictions = []
            for A in train_matrices:
                predictions.append(max(int(A @ t0_concentration), 0))
            pred = round(np.mean(predictions), 1)
            logging.info(f'Ensenmble CDR prediction for subject {subj}: {pred}')
            ae = abs(t1_cdr - pred)
            #pcc = pearson_corr_coef(t1_cdr, pred)[0]
            if np.isnan(ae) or np.isinf(ae): raise Exception("Invalid value of ABSERR")
            #if np.isnan(pcc): raise Exception("Invalid value of PCC")
            #np.savetxt(subj + 'test/CDRMR_test_' + date + '.csv', A_train, delimiter=',')
            #save_prediction_plot(t0_concentration, pred, t1_cdr, subj, subj +'test/CDRMR_test_' + date + '.png', ae, pcc)
        except Exception as e:
            logging.error(e)
            continue
        else:
            ae_subj.append(ae)
            test_scores[subj] = [ae, pred, t1_cdr]

    total_ae.append(np.mean(ae_subj, axis=0))

    return


if __name__ == '__main__':

    ### INPUT ###
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            print("Using default")
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    if not os.path.exists(output_mat):
        os.makedirs(output_mat)
    if not os.path.exists(output_res):
        os.makedirs(output_res)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            print("Using default")
            num_cores = cpu_count()
            logging.info(f"{num_cores} cores available")

    train_size = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    while train_size <= 0 or train_size > len(dataset.keys())-1:
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())-1}, default]: '))
        except Exception as e:
            print("Using default")
            train_size = len(dataset.keys())-1
    test_size = len(dataset.keys()) - train_size

    iter_max = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 3\'000\'000]: '))
        except Exception as e:
            print("Using default")
            iter_max = 3e6

    N_fold = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while N_fold < 1:
        try:
            N_fold = int(input(f'Folds for cross validation [default {len(dataset.keys())}]: '))
            if N_fold > len(dataset.keys()) or N_fold > (len(dataset.keys()) - test_size + 1): 
                logging.raiseExceptions("Invalid number of folds")
                raise Exception("Number of folds is greater than number of subjects or number of possible splits")
        except Exception as e:
            print("Using default")
            N_fold = len(dataset.keys())

    ### SIMULTATIONS ###

    training_scores = {}
    test_scores = {}
    total_ae = []
    total_pcc = []
    total_reg_err = []

    total_time = time()
    train_time = 0

    lock = Lock()

    A_matrices = {}

    for i in tqdm(range(N_fold)):   
        train_set = {}
        test_set = {}
        
        counter = 0
        for k in dataset.keys():
            # NOTE: dataset keys are subjects paths
            if not os.path.exists(k+'train/'):
                os.makedirs(k+'train/')
            if not os.path.exists(k+'test/'):
                os.makedirs(k+'test/')
            if counter >= i and counter < test_size+i:
                test_set[k] = dataset[k]
            else:
                train_set[k] = dataset[k]
            counter += 1    
        
        start_time = time()
        train_matrices = training()
        train_time += time() - start_time

        test(train_matrices)
        logging.info(f"*** Fold {i} completed ***")

    ### OUTPUT STATS ###
    np.savetxt(f"{output_mat}CDRMR_{category}_{date}.csv", np.mean(np.array(list(A_matrices.values())), axis=0), delimiter=',')
    #np.savetxt(f"{output_mat}CDRMR_{category}_regions_{date}.csv", np.mean(np.array(total_reg_err), axis=0), delimiter=',')

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg AE test", "SD AE test"]
    pt_avg.add_row([round(np.mean(total_ae), digits), round(np.std(total_ae), 2)])
        
    pt_train = PrettyTable()
    pt_train.field_names = ["ID", "Avg AE train"]
    pt_train.sortby = "ID"

    for s in training_scores.keys():
        ae_subj = training_scores[s]
        pt_train.add_row([s, round(ae_subj, digits)])

    pt_test = PrettyTable()
    pt_test.field_names = ["ID", "Avg AE test", "Assigned CDR", "True CDR"]
    pt_test.sortby = "ID"

    for s in test_scores.keys():
        ae_subj = test_scores[s][0]
        pt_test.add_row([s, round(np.mean(ae_subj), digits), test_scores[s][1], test_scores[s][2]])

    total_time = time() - total_time
    filename = f"{output_res}CDRMR_{category}_{date}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Iterations per patient: {iter_max}\n")
    out_file.write(f"Folds: {N_fold}\n")
    out_file.write(f"Elapsed time for training (s): {format(train_time, '.2f')}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string() + '\n')
    out_file.write(pt_train.get_string() + '\n')
    out_file.write(pt_test.get_string() + '\n')
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info(f"Training set size: {train_size}")
    logging.info(f"Testing set size: {len(dataset.keys())-train_size}")
    logging.info(f"Iterations per patient: {iter_max}")
    logging.info(f"Folds: {N_fold}")
    logging.info(f"Elapsed time for training (s): {format(train_time, '.2f')}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")
