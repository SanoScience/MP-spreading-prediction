"""
    SYNOPSIS
    python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <matrix> <use_binary> <iter_max> <N_fold>
"""

''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

from email.policy import default
import os
import logging
import sys
from time import time
import json
from tqdm import tqdm 
import numpy as np
from scipy.stats import pearsonr as pearson_corr_coef
from utils_vis import *
from utils import *
from datetime import datetime
import multiprocessing
from prettytable import PrettyTable
import yaml
from sklearn.metrics import mean_squared_error

np.seterr(all = 'raise')
date = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_MAR_{date}.log")

class MARsimulation:
    def __init__(self, subject, connect_matrix, t0_concentrations, t1_concentrations, lam, matrix, use_binary, iter_max=int(3e6)):
        self.subject = subject
        self.cm = connect_matrix
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations
        self.lam = lam
        self.matrix = matrix
        self.use_binary = use_binary
        self.iter_max = iter_max

        self.error_stop = 1e-5                                                  # acceptable error threshold for the reconstruction error
        self.gradient_thr = 1e-20
        self.eta = 1e-5                                             
        self.max_retry = 10 
        self.max_bad_iter = 1

    def run(self):        
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
        constraint the use of existing anatomical connections (this is **B** in our paper).
        B has zero elements where no structural connectivity appears (the CM has a diagonal set manually to 0, so we restore it to one).  '''
        if self.use_binary:
            self.B = np.where(self.cm>0, 1, 0) + np.identity(self.cm.shape[0])
        else:
            self.B = np.ones_like(self.cm)
    
    def generate_A(self):
        ''' Generate the matrix A from which starting the optimization (this is **A** in our paper). '''
        if self.matrix == 0: # CM (assuming it's diagonal has been set to zero by normalization)
            return np.copy(self.cm) + np.identity(self.cm.shape[0])
        elif self.matrix == 1: # Random
            return np.random.rand(self.cm.shape[0], self.cm.shape[1])
        elif self.matrix == 2: # diagonal
            return np.diag(self.init_concentrations)
        else: # identity
            return np.identity(self.cm.shape[0])

    def run_gradient_descent(self, vis_error=False):                           
        error_reconstruct = 1 + self.error_stop         
        previous_error = -1             

        # reconstruction error along iterations (for debugging purposes)           
        if vis_error: error_buffer = [] 
        trial = 0
        
        while trial<self.max_retry:
            iter_count = 0     
            bad_iter_count = 0                          
            A = self.generate_A()
            while (error_reconstruct > self.error_stop) and iter_count < self.iter_max:
                try:
                    # NOTE: numpy.linalg.norm has a parameter 'ord' which specifies the kind of norm to compute
                    # ord=1 corresponds to max(sum(abs(x), axis=0))
                    # ord=None corresponds to Frobenius norm (square root of the sum of the squared elements)

                    # calculate reconstruction error                    
                    error_reconstruct = 0.5 * np.linalg.norm(self.final_concentrations - (A * self.B) @ self.init_concentrations, ord=2)**2
                    if vis_error: error_buffer.append(error_reconstruct)
            
                    if previous_error != -1 and previous_error <= error_reconstruct:
                        bad_iter_count += 1 
                        if bad_iter_count == self.max_bad_iter:
                            logging.info("Error is raising, halting optimization")
                            break
                    else:
                        bad_iter_count = 0
                    previous_error = error_reconstruct
                    if vis_error: error_buffer.append(error_reconstruct)
                except FloatingPointError as e:
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during computation of reconstruction error (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    gradient = (-(self.final_concentrations - (A * self.B) @ self.init_concentrations) @ self.init_concentrations.T) * self.B
                    norm = np.linalg.norm(gradient)
                    if norm <= self.gradient_thr:
                        logging.info(f"Gradient for subject {self.subject} met termination criterion")
                        break
                    
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during gradient computation (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break

                try:
                    A = A - (self.eta * gradient)
                except FloatingPointError as e:   
                    logging.warning(e)
                    logging.warning(f'Overflow encountered during updating of coefficient matrix (iteration {iter_count}) for subject {self.subject}. Restarting with smaller steps ({trial}/{self.max_retry})')
                    break
                
                iter_count += 1
                
            if error_reconstruct <= self.error_stop or bad_iter_count == self.max_bad_iter or iter_count == self.iter_max or norm <= self.gradient_thr: break
            trial += 1
            self.eta /= 10 
        
        if trial == self.max_retry: logging.error(f"Subject {self.subject} couldn't complete gradient descent")           
        if vis_error: visualize_error(error_buffer)
        logging.info(f"Final reconstruction error for subject {self.subject}: {error_reconstruct} ({iter_count} iterations)")
        return A
                  
def run_simulation(subject, paths, connect_matrix, lam, matrix, use_binary, iter_max, queue):    
    ''' Run simulation for single patient. '''
    
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['CM']))
        t0_concentration = load_matrix(paths['baseline']) 
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        logging.error(f"Exception during loading of CM/ for subject {subject}")
        return

    mse = pcc = None
    try:
        simulation = MARsimulation(subject, connect_matrix, t0_concentration, t1_concentration, lam, matrix, use_binary, iter_max)
    except Exception as e:
        logging.error(f"Exception happened during initialization of 'MARsimulation' object for subject {subject}. Traceback:\n{e}") 
        return

    try:
        logging.info(f"Starting gradient descent for subject {subject}")
        t1_concentration_pred = drop_negative_predictions(simulation.run())
    except Exception as e:
        logging.error(f"Exception happened for during execution of simulation for subject {subject}. Traceback:\n{e}") 
        return 

    try:
        logging.info(f"Computiong PCC and MSE for subject {subject}")
        mse = mean_squared_error(t1_concentration, t1_concentration_pred)
        pcc = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
        if np.isnan(pcc): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(f"Exception happened for Pearson correlation coefficient and MSE computation for subject {subject}. Traceback:\n{e}") 
        return
    
    try:
        logging.info(f"Saving prediction plot for subject {subject}")
        save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subject, subject + 'train/MAR_train_' + date + '.png', mse, pcc)
        save_coeff_matrix(subject + 'train/MAR_train_' + date + '.csv', simulation.coef_matrix)
        
    except Exception as e:
        logging.error(f"Exception happened during saving of simulation results for subject {subject}. Traceback:\n{e}") 
        return

    queue.put([subject, mse, pcc])

    return      

def training(train_set, num_cores, lam, matrix, use_binary, iter_max, training_scores):
    procs = []
    queue = multiprocessing.Queue()
    
    for subj, paths in train_set.items():
        p = multiprocessing.Process(target=run_simulation, args=(
                                                                    subj, 
                                                                    paths, 
                                                                    None, 
                                                                    lam, 
                                                                    matrix, 
                                                                    use_binary, 
                                                                    iter_max, 
                                                                    queue
                                                                ))
        p.start()
        procs.append(p)  
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)
                    break
    for p in procs:
        p.join()

    while not queue.empty():
        subj, mse, pcc = queue.get()
        # mse and pcc are -1 if an exception happened during simulation, and are therefore not considered
        training_scores[subj] = [mse, pcc]
    
    coeff_matrices = []
    # read results saved by "run_simulation method"
    for subj, _ in train_set.items():
        try:
            coeff_matrices.append(load_matrix(subj + 'train/MAR_train_' + date + '.csv'))
        except Exception as e:
            logging.error(f"Matrix for subject {subj} with a suitable date could not be found")

    avg_coeff_matrix = np.mean(coeff_matrices, axis=0)
    return avg_coeff_matrix

def test(conn_matrix, test_set, test_scores):
    mse_subj = []
    pcc_subj = []
    reg_err = []
    for subj, paths in test_set.items():
        logging.info(f"Test on subject {subj}")
        try:
            t0_concentration = load_matrix(paths['baseline'])
            t1_concentration = load_matrix(paths['followup'])
            pred = conn_matrix @ t0_concentration
            mse = mean_squared_error(t1_concentration, pred)
            pcc = pearson_corr_coef(t1_concentration, pred)[0]
            if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
            save_prediction_plot(t0_concentration, pred, t1_concentration, subj, subj +'test/MAR_test_' + date + '.png', mse, pcc)
            regional_error = np.abs(pred - t1_concentration)
            save_coeff_matrix(subj + 'test/MAR_test_regions_' + date + '.csv', regional_error)
        except Exception as e:
            logging.error(e)
            continue
        else:
            mse_subj.append(mse)
            pcc_subj.append(pcc)
            reg_err.append(regional_error)
            test_scores[subj] = [mse, pcc]
    
    avg_mse = np.mean(mse_subj, axis=0)
    avg_pcc = np.mean(pcc_subj, axis=0)
    avg_reg_err = np.mean(reg_err, axis=0)

    return avg_mse, avg_pcc, avg_reg_err


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
            num_cores = multiprocessing.cpu_count()
            logging.info(f"{num_cores} cores available")

    train_size = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    while train_size <= 0 or train_size > len(dataset.keys())-1:
        try:
            train_size = int(input(f'Number of training samples [max {len(dataset.keys())-1}, default]: '))
        except Exception as e:
            print("Using default")
            train_size = len(dataset.keys())-1
    test_size = len(dataset.keys()) - train_size
            
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while lam < 0 or lam > 1:
        try:
            lam = float(input('Insert the lambda coefficient for L1 penalty [0..1, default 0]: '))
        except Exception as e:
            print("Using default")
            lam = 0
    
    matrix = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while matrix < 0 or matrix > 3:
        try:
            matrix = int(input('Choose the initial A matrix [0=CM, 1=Rnd, 2=Diag, 3=Identity default 0]: '))
        except Exception as e:
            print("Using default")
            matrix = 0

    use_binary = (True if sys.argv[6]=='1' else False) if len(sys.argv) > 6 else -1
    while use_binary < 0:
        try:
            use_binary = True if input('Choose if using binary matrix [1] or not [0, default]: ') == '1' else False
        except Exception as e:
            print("Using default")
            use_binary = False

    iter_max = int(sys.argv[7]) if len(sys.argv) > 7 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 3\'000\'000]: '))
        except Exception as e:
            print("Using default")
            iter_max = 3e6

    N_fold = int(sys.argv[8]) if len(sys.argv) > 8 else -1
    while N_fold < 1:
        try:
            N_fold = int(input(f'Folds for cross validation [default {len(dataset.keys())}]: '))
            if N_fold > len(dataset.keys()) or N_fold > len(dataset.keys())/train_size: 
                logging.raiseExceptions("Invalid number of folds")
                raise Exception("Number of folds is greater than number of subjects or number of possible splits")
        except Exception as e:
            print("Using default")
            N_fold = len(dataset.keys())

    ### SIMULTATIONS ###

    training_scores = {}
    test_scores = {}
    total_mse = []
    total_pcc = []
    total_reg_err = []

    total_time = time()
    train_time = 0

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
        coeff_matrix = training(train_set, num_cores, lam, matrix, use_binary, iter_max, training_scores)
        train_time += time() - start_time

        mse, pcc, reg_err = test(coeff_matrix, test_set, test_scores)
        total_mse.append(mse)
        total_pcc.append(pcc)
        total_reg_err.append(reg_err)
        logging.info(f"*** Fold {i} completed ***")

    ### OUTPUT STATS ###

    # NOTE: the coefficient matrix is saved from the last fold
    np.savetxt(f"{output_mat}MAR_{category}_{date}.csv", coeff_matrix, delimiter=',')
    np.savetxt(f"{output_mat}MAR_{category}_regions_{date}.csv", np.mean(np.array(total_reg_err), axis=0), delimiter=',')

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg MSE test", "SD MSE test", "Avg Pearson test", "SD Pearson test"]
    pt_avg.add_row([round(np.mean(total_mse), 5), round(np.std(total_mse), 2), round(np.mean(total_pcc), 5), round(np.std(total_pcc), 2)])
        
    pt_train = PrettyTable()
    pt_train.field_names = ["ID", "Avg MSE train", "SD MSE train", "Avg Pearson train", "SD Pearson train"]
    pt_train.sortby = "ID"

    for s in training_scores.keys():
        mse_subj = [training_scores[s][0]]
        pcc_subj = [training_scores[s][1]]
        pt_train.add_row([s, round(np.mean(mse_subj), 5), round(np.std(mse_subj), 2), round(np.mean(pcc_subj), 5), round(np.std(pcc_subj), 2)])

    pt_test = PrettyTable()
    pt_test.field_names = ["ID", "Avg MSE test", "SD MSE test", "Avg Pearson test", "SD Pearson test"]
    pt_test.sortby = "ID"

    for s in test_scores.keys():
        mse_subj = [test_scores[s][0]]
        pcc_subj = [test_scores[s][1]]
        pt_test.add_row([s, round(np.mean(mse_subj), 5), round(np.std(mse_subj), 2), round(np.mean(pcc_subj), 5), round(np.std(pcc_subj), 2)])

    total_time = time() - total_time
    filename = f"{output_res}MAR_{category}_{date}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Lambda coefficient: {lam}\n")
    out_file.write(f"Matrix: {'CM' if matrix==0 else 'Rnd' if matrix==1 else 'Diag'}\n")
    out_file.write(f"Binary matrix: {'yes' if use_binary else 'no'}\n")
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
    logging.info(f"Lambda coefficient: {lam}")
    logging.info(f"Matrix: {'CM' if matrix==0 else 'Rnd' if matrix==1 else 'Diag'}")
    logging.info(f"Binary matrix: {'yes' if use_binary else 'no'}")
    logging.info(f"Iterations per patient: {iter_max}")
    logging.info(f"Folds: {N_fold}")
    logging.info(f"Elapsed time for training (s): {format(train_time, '.2f')}")
    logging.info(f"Total time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")

    quit()