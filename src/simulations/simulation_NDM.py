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
from multiprocessing import cpu_count
from threading import Thread, active_count, Lock
import os
import logging
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
import warnings

np.seterr(all = 'raise')
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.ERROR, force=True, filename = f"trace_NDM_{date}.log")
digits = 5

class NDM(Thread):
    def __init__(self, subject, paths, downsample = False):
        Thread.__init__(self)
        ''' 
        t_total: gap between baseline and followup PET (in years)
            timestep = t_total / (iterations * 10)
            iterations = t_total / (timestep * 10)
            t_total = timestep * iterations * 10
        Putting iterations to '1' and for t_total = 2 years, the timestep is 0.2
        '''
        self.subject = subject
        self.paths = paths

        self.downsample = downsample
        
        self.t_total = 2 # total length of the simulation in years
        self.timestep = 1e-5
        self.iterations = 2318 # number of iterations obtained through empirical test (2318)

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
        # Used to suppress a FutureWarning about return of normalized_laplacian_matrix() in future versions
        warnings.filterwarnings("ignore")
        self.cm = np.asmatrix(self.cm) + np.identity(self.cm.shape[0])
        G = nx.from_numpy_matrix(self.cm)
        self.L = nx.normalized_laplacian_matrix(G).toarray()
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        self.eigvecs = self.eigvecs.real
        self.eigvals = np.array(self.eigvals).real # Taking only the real part
        self.inv_eigvecs = np.linalg.inv(self.eigvecs + eps)
                   

    def integration_step(self):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        step = 0
        try:
            exp = np.exp(beta * self.eigvals * -1)
            step = self.eigvecs * exp * self.eigvecs @  self.diffusion[-1]
        except Exception as e:
            logging.error("Error during integration step")
            logging.error(e)
            print("Error during integration step")
            print(e)
        self.diffusion.append(self.diffusion[-1] + step * self.timestep)
        return
    
    def iterate_spreading(self):
        self.diffusion = [self.t0_concentration]  
        for _ in range(self.iterations):
            try:
                self.integration_step()
            except Exception as e:
                logging.error(f"Error during iterate spreading")
                logging.error(e)
                print(f"Error during integration step")
                print(e)
                break 
        
        return np.asarray(self.diffusion[-1], dtype=object)
 
    def downsample_matrix(self, matrix, target_len=int(1e3)):
        ''' Take every n-th sample when the matrix is longer than target length. '''
        current_len = matrix.shape[0]
        if current_len > target_len:
            factor = int(current_len/target_len)
            matrix = matrix[::factor, :] # downsampling
        return matrix

    def run(self):    
        
        if not os.path.exists(self.subject+'test/'):
            os.makedirs(self.subject+'test/')    

        try:
            self.cm = drop_data_in_connect_matrix(load_matrix(paths['CM']))
            #connect_matrix = prepare_cm(connect_matrix)
            self.t0_concentration = load_matrix(paths['baseline'])
            self.t1_concentration = load_matrix(paths['followup'])
            self.calc_laplacian()  
        except Exception as e:
            logging.error(f"Error during data load for subject {self.subject}. Traceback: ")
            logging.error(e)
            print(f"Error during data load for subject {self.subject}. Traceback: ")
            print(e)
            return

        try:
            t1_concentration_pred = self.iterate_spreading()
            if self.downsample: 
                self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        except Exception as e:
            logging.error(f"Error during simulation for subject {self.subject}. Traceback: ")
            logging.error(e)
            print(f"Error during simulation for subject {self.subject}. Traceback: ")
            print(e)
            return
        
        try:
            mse = mean_squared_error(self.t1_concentration, t1_concentration_pred)
            pcc = pearson_corr_coef(self.t1_concentration, t1_concentration_pred)[0]
            reg_err = np.abs(t1_concentration_pred - self.t1_concentration)
            if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
            if np.isnan(pcc): raise Exception("Invalid value of PCC")
        except Exception as e:
            logging.error(f"Error during computation of statistics for subject {self.subject}. Traceback: ")
            logging.error(e)
            print(f"Error during computation of statistics for subject {self.subject}. Traceback: ")
            print(e)
            return
        
        logging.info(f"Saving prediction for subject {self.subject}")
        try:
            np.savetxt(os.path.join(self.subject, 'test/NDM_diffusion_' + date + '.csv'), self.diffusion, delimiter=',')
            np.savetxt(os.path.join(self.subject, 'test/NDM_terminal_concentrations_' + date + '.csv'), t1_concentration_pred, delimiter=',')
        except Exception as e:
            logging.error(f"Error during save of prediction for subject {self.subject}. Traceback: ")
            logging.error(e)
            print(f"Error during save of prediction for subject {self.subject}. Traceback: ")
            print(e)
            return
        
        lock.acquire()
        save_prediction_plot(self.t0_concentration, t1_concentration_pred, self.t1_concentration, self.subject, os.path.join(self.subject, 'test/NDM_' + date + '.png'), mse, pcc)
        mse_list.append(mse)
        pcc_list.append(pcc)
        reg_err_list.append(reg_err)
        pt_subs.add_row([self.subject, round(mse,digits), round(pcc,digits)])
        lock.release()

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
            print('Using default')
            category = 'ALL'
        category = 'ALL' if category == '' else category
        
    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)
    if not os.path.exists(output_mat):
        os.makedirs(output_mat)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    while num_cores < 1:
        try:
            num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
        except Exception as e:
            print('Using default')
            num_cores = cpu_count()
            logging.info(f"{num_cores} cores available")
            
    beta = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while beta < 0: 
        try:
            beta = float(input('Insert the beta value [0.1 by default]: '))
        except Exception as e:
            print('Using default')
            beta = 0.1

    ### SIMULATIONS ###

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "MSE", "Pearson"]
    pt_subs.sortby = "ID" # Set the table always sorted by patient ID

    mse_list = []
    pcc_list = []
    reg_err_list = []
    
    total_time = time()
    
    lock = Lock()
    works = []
    for subj, paths in dataset.items():
        works.append(NDM(subj, paths))
        works[-1].start()

        while len(works) >= num_cores:
            for w in works:
                if not w.is_alive():
                    works.remove(w)
                    break

    for w in works:
        w.join()
        works.remove(w)
        
    
    ### OUTPUTS ###

    np.savetxt(f"{output_mat}NDM_{category}_regions_{date}.csv", np.mean(np.array(reg_err_list), axis=0), delimiter=',')
    pt_avg.add_row([round(np.mean(mse_list, axis=0), digits), round(np.std(mse_list, axis=0), 2), round(np.mean(pcc_list, axis=0), digits), round(np.std(pcc_list, axis=0), 2)])

    total_time = time() - total_time
    filename = f"{output_res}/NDM_{category}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
    out_file= open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Beta: {beta}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Elapsed time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())
    out_file.close()
    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Beta: {beta}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info(f"Elapsed time (s): {format(total_time, '.2f')}")
    logging.info('***********************')
    logging.info(f"Results saved in {filename}")
    print(f"Results saved in {filename}")