''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

Based on publication: 
"Epidemic Spreading Model to Characterize Misfolded Proteins Propagation 
in Aging and Associated Neurodegenerative Disorders"
Authors: Yasser Iturria-Medina ,Roberto C. Sotero,Paule J. Toussaint,Alan C. Evans 
'''

from datetime import datetime
from glob import glob
import json
import multiprocessing 
import os
import logging
import random
import sys
from time import time
from prettytable import PrettyTable

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import _dijkstra
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import drop_data_in_connect_matrix, load_matrix, calc_rmse, calc_msle, save_terminal_concentration

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.DEBUG)

class EMS_Simulation:
    ''' A class to simulate the spread of misfolded beta_amyloid. '''

    def __init__(self, connect_matrix, regions_distances, years, concentrations=None, iter_max=10000, beta0=0.1):
        self.N_regions = 166    # number of brain regions 
        self.speed = 1          # propagation velocity
        self.dt = 0.0001          # time step (keep it very small to avoid overflows)
        self.T_total = years    # total time 
        self.amy_control = 3    # parameter to control the number of voxels in which beta-amyloid may get synthesized (?)
        self.prob_stay = [0.5 for _ in range(166)]    # the probability of staying in the same region per unit time (0.5)
        self.trans_rate = 116     # a scalar value, controlling the baseline infectivity
        self.iter_max = iter_max      # max no. of iterations
        self.mu_noise = 0       # mean of the additive noise
        self.sigma_noise = 1    # standard deviation of the additive noise
        self.gini_coeff = 1     # measure of statistical dispersion in a given system, with value 0 reflecting perfect equality and value 1 corresponding to a complete inequality
        self.clearance_rate = np.ones(self.N_regions)   
        self.synthesis_rate = np.ones(self.N_regions)  
        self.beta0 = beta0
        
        if concentrations is not None: 
            #logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            #logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
        
        self.regions_distances = regions_distances
        self.connect_matrix = connect_matrix #+ np.where(connect_matrix > 0, 1e-2, 0)  # add noise to connectivity matrix
        
    def define_seeds(self, init_concentration=0.2):
        ''' Define Alzheimer seed regions manually. 
        
        Args:
            init_concentration (int): initial concentration of misfolded proteins in the seeds. '''
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.N_regions)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        # assign initial concentration of proteins in this region
        diffusion_init[[31, 32, 35, 36]] = init_concentration
        return diffusion_init
    
    def remove_diagonal(self, matrix):
        # remove diagonal elements from matrix
        matrix -= np.diag(np.diag(matrix))
        return matrix

    def calculate_connect(self):
        eps = 1e-2
        self.regions_distances = self.remove_diagonal(self.regions_distances)
        self.regions_distances += eps # add small epsilon to avoid dividing by zero
        self.connect_matrix = self.remove_diagonal(self.connect_matrix)
        
        connect = self.connect_matrix
        noise = np.random.normal(self.mu_noise, self.sigma_noise, (self.N_regions, self.N_regions))     
       
        Epsilon = np.zeros((self.N_regions, self.N_regions))
        
        for i in range(self.N_regions):
            beta = 1 - np.exp(-self.beta0 * self.prob_stay[i])  # regional probability receiving MP infectous-like agents
            t = 0
            for j in range(self.N_regions):
                if i != j:
                    t +=  beta * self.prob_stay[i] * (self.connect_matrix[i, j] * self.gini_coeff + self.connect_matrix[i, i] * (1 - self.gini_coeff))
            Epsilon[i] = t
            
        # INTRA-BRAIN EPIDEMIC SPREADING MODEL 
        connect = (1 - self.prob_stay[i])*Epsilon - self.prob_stay * np.exp(- self.diffusion_init* self.prob_stay) +  noise
            
        # The probability of moving from region i to edge (i,j)
        sum_line = np.sum(connect, axis=1)
        connect /= sum_line
        return connect
    
    def calculate_Rnor_Pnor(self, connect):  
        ''' Calculate and return no. of normal amyloid in regions and paths. '''
                   
        Rnor = np.zeros(self.N_regions) # number of  normal amyloid in regions
        Pnor = np.zeros(self.N_regions) # number of  normal amyloid in paths

        movOut = np.zeros((self.N_regions, self.N_regions))

        # moving process
        for _ in range(self.iter_max):  
            # movDrt stores the number of proteins towards each region. 
            # i.e. element in kth row lth col denotes the number of proteins in region k moving towards l
            movDrt = Rnor * connect * self.dt
            movDrt = self.remove_diagonal(movDrt)
        
            # paths towards regions
            # longer path & smaller v = lower probability of moving out of paths
            # update moving
            movOut = (self.speed * Pnor)  / self.regions_distances
            movOut = self.remove_diagonal(movOut)

            Pnor = Pnor - movOut * self.dt + movDrt
            Pnor = self.remove_diagonal(Pnor)
            Sum_rows_movOut = np.sum(movOut, axis=1)
            Sum_cols_movDrt = np.sum(movDrt, axis=0)

            Rtmp = Rnor
            Rnor = Rnor +  np.transpose(Sum_rows_movOut) * self.dt -  Sum_cols_movDrt

            #growth process	
            precision = 1e-15
            Rnor = Rnor -  Rnor * (1 - np.exp(- self.clearance_rate * self.dt)) + (self.synthesis_rate * self.amy_control) * self.dt
            if abs(Rnor - Rtmp).all() < (precision * Rtmp).all():
                break
    
        return Rnor, Pnor

    def calculate_Rmis_Pmis(self, connect, Rnor0, Pnor0):
        ''' Calculate and return no. of misfolded amyloid in regions and paths. 
        at each timestep. 
        
        Args:
            Rnor0 (list): vector with length N_Regions, the population of normal agents in regions before pathogenic spreading 
            Pnor0 (list): vector with length N_Regions, the population of normal agents in edges before pathogenic spreading 
        '''

        # store results for single timepoint
        Rmis = np.zeros(self.N_regions)                         # number of misfolded beta-amyloid in regions
        Pmis = np.zeros((self.N_regions, self.N_regions))       # number of misfolded beta-amyloid in paths

        # store results at each timepoint
        Rmis_all = np.zeros((self.N_regions, self.T_total))
        Pmis_all = np.zeros((self.N_regions, self.N_regions, self.T_total))
        
        # misfolded protein spreading process
        movOut_mis = np.zeros((self.N_regions, self.N_regions))
        movDrt_mis = np.zeros((self.N_regions, self.N_regions))

        # fill with PET data 
        Rmis = self.diffusion_init
        
        for t in range(self.T_total):
            # moving process
            # misfolded proteins: region -->> paths
            movDrt_mis = Rnor0 * connect * self.dt
            movDrt_mis = self.remove_diagonal(movDrt_mis)
            
            # normal proteins: paths -->> regions
            movOut_mis = (Pnor0 / self.regions_distances)* self.speed
            movOut_mis = self.remove_diagonal(movOut_mis)
        
            # update regions and paths
            Pmis = Pmis - movOut_mis * self.dt +  movDrt_mis
            Pmis = self.remove_diagonal(Pmis)

            Sum_rows_movOut_mis = np.sum(movOut_mis, axis=1)
            Sum_cols_movDrt_mis = np.sum(movDrt_mis, axis=0)

            Rmis = Rmis + np.transpose(Sum_rows_movOut_mis) * self.dt  - Sum_cols_movDrt_mis
            Rmis_cleared = Rmis * (1 - np.exp(-self.clearance_rate * self.dt))
    
            # probability of getting misfolded
            misProb = 1 - np.exp( - Rmis * self.beta0 * self.dt)

            # number of newly infected
            N_misfolded = Rnor0 * np.exp(- self.clearance_rate) * misProb 
    
            # update
            Rmis = Rmis - Rmis_cleared + N_misfolded + (self.synthesis_rate * self.amy_control) * self.dt
            
            Rmis_all[:,t] = Rmis # regions
            Pmis_all[:, :, t] = Pmis # paths
            
        return Rmis_all, Pmis_all

def run_simulation(paths, output_dir, subject, iter_max, beta_iter, beta0=0.1, queue = None):    
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

    years = 50
    
    regions_distances = dijkstra(connect_matrix)
    beta_step = 0.0001
    opt_beta0 = opt_rmse = opt_pcc = -1
    for _ in range(beta_iter):
        simulation = EMS_Simulation(connect_matrix, regions_distances, years, concentrations=t0_concentration, iter_max=iter_max, beta0=beta0)
        beta0 += beta_step
        connect = simulation.calculate_connect()
        Rnor, Pnor = simulation.calculate_Rnor_Pnor(connect)
        Rmis_all, Pmis_all = simulation.calculate_Rmis_Pmis(connect, Rnor, Pnor)

        #visualize_diffusion_timeplot(Rmis_all, simulation.dt, simulation.T_total)

        # predicted vs real plot; take results from the last step
        t1_concentration_pred = Rmis_all[:, years-1]
        try:
            rmse = calc_rmse(t1_concentration, t1_concentration_pred)
            corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
            assert rmse != 'nan' and rmse != 'inf'
            assert corr_coef != 'nan' and corr_coef != 'inf'
        except Exception as e:
            logging.error(e)
            continue
        else:
            if opt_rmse == -1 or rmse < opt_rmse:
                opt_rmse = rmse 
                opt_pcc = corr_coef
                opt_beta0 = beta0
        '''
        visualize_terminal_state_comparison(t0_concentration, 
                                            t1_concentration_pred, 
                                            t1_concentration, 
                                            subject,
                                            rmse,
                                            corr_coef)
        '''
        #save_terminal_concentration(subject_output_dir, t1_concentration_pred, 'EMS')
    if queue:
        queue.put([opt_beta0, opt_rmse, opt_pcc])
    
    return


def dijkstra(matrix):
    # calculate distance matrix using Dijkstra algorithm 
    # in order to find the shortest path between nodes in a graph
    L = len(matrix)
    distance = np.zeros((L,L))
    G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
    for i in range(L):
        for j in range(L):
            t = nx.has_path(G,i,j)
            if t == False:
                distance[i,j] = 0
            else:
                t = nx.dijkstra_path_length(G, i, j, weight = 'weight')
                distance [i, j] = t
                
    return distance

if __name__=="__main__":
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
    pt_avg.field_names = ["Avg RMSE", "SD RMSE", "Avg Pearson", "SD Pearson"]

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

    iter_max = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    while iter_max <= 0:
        try:
            iter_max = int(input('Insert the maximum number of iterations [hit \'Enter\' for 10\'000]: '))
        except Exception as e:
            iter_max = 10000

    beta_iter = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    while beta_iter <= 0 :
        try:
            beta_iter = int(input('Insert the number of iterations for beta0 estimation: '))
        except Exception as e:
            logging.error(e)

    N_fold = int(sys.argv[6]) if len(sys.argv) > 6 else -1
    while N_fold < 1:
        try:
            N_fold = int(input('Folds for cross validation: '))
        except Exception as e:
            logging.error(e)
            continue

    procs = []
    queue = multiprocessing.Queue()
    train_time = 0
    train_b0s = []
    total_rmse = []
    total_pcc = []
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
        for subj, paths in train_set.items():
            p = multiprocessing.Process(target=run_simulation, args=(paths, output_dir, subj, iter_max, beta_iter, 0.1, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
        for p in procs:
            p.join()

        beta0_list = []
        while not queue.empty():
            beta0, _, _ = queue.get()
            beta0_list.append(beta0)
        
        avg_beta0 = np.mean(beta0_list, axis=0)
        train_b0s.append(avg_beta0)

        train_time += time() - start_time

        for subj, paths in test_set.items():
            p = multiprocessing.Process(target=run_simulation, args=(paths, output_dir, subj, iter_max, 1, avg_beta0, queue))
            p.start()
            procs.append(p)

            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    if not p.is_alive():
                        procs.remove(p)
        for p in procs:
            p.join()

        test_rmse_list = []
        test_pcc_list = []
        while not queue.empty():
            _, rmse, pcc = queue.get()
            test_rmse_list.append(rmse)
            test_pcc_list.append(pcc)
        
        total_rmse.append(np.mean(test_rmse_list, axis=0))
        total_pcc.append(np.mean(test_pcc_list, axis=0))
   
    pt_avg.add_row([format(np.mean(total_rmse, axis=0), '.2f'), format(np.std(total_rmse, axis=0), '.2f'), format(np.mean(total_pcc, axis=0), '.2f'), format(np.std(total_pcc, axis=0), '.2f')])
    filename = f"../../results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_EMS_{category}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Training set size: {train_size}\n")
    out_file.write(f"Testing set size: {len(dataset.keys())-train_size}\n")
    out_file.write(f"Iterations per patient: {iter_max}\n")
    out_file.write(f"Iterations per beta0: {beta_iter}\n")
    out_file.write(f"Folds: {N_fold}\n")
    out_file.write(f"Elapsed time for training (s): {format(train_time, '.2f')}\n")
    out_file.write(f"Average b0: {np.mean(train_b0s, axis=0)}\n")
    out_file.write(pt_avg.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")