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
import warnings
from prettytable import PrettyTable

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import _dijkstra
from scipy.stats.stats import pearsonr as pearson_corr_coef

from scipy.stats import norm
from scipy.stats import zscore

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import drop_data_in_connect_matrix, load_matrix, calc_rmse, calc_msle, save_terminal_concentration

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.DEBUG)

# Distance matrix
def dijkstra(matrix):
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

def drop_negative_predictions(predictions):
    return np.maximum(predictions, 0)

def compute_gini(concentration):
    '''
    https://en.wikipedia.org/wiki/Gini_coefficient#Definition
    Compute Gini coefficient as half of the relative mean absolute difference, which is mathematically equivalent to the definition based on the Lorenz curve.
    The mean absolute difference is the average absolute difference of all pairs of items of the population, and the relative mean absolute difference is the mean absolute difference divided by the average to normalize for scale.
    '''
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(concentration, concentration)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(concentration)
    # Gini coefficient
    return 0.5 * rmad

'''
def define_initial_P(concentration, connect_matrix, distances, max_concentration):
    # NOTE: both connectivity and distances matrix have 0 diagonals (self connections are not considered)
    
    # I consider only connections with infective regions (I am cutting out rows corresponding to regions without Amyloid)
    # I am expanding the binary vector to a binary matrix to apply it to the connectivity matrix
    infected_regions = np.tile(np.where(concentration>0, 1, 0), (np.shape(connect_matrix)[0],1)) * connect_matrix
    
    # NOTE: missing connections and not infective regions have value 0 in 'infected_regions' matrix
    P = 1 - np.exp(- np.sum.outer(infected_regions, infected_regions)/np.sum.outer(distances))
    
    return P
'''

def Simulation(concentration, connect_matrix, years, timestep, beta_0, delta_0, mu_noise, sigma_noise, velocity=1, n_regions = 166):
    
    #TODO: should I use something different from 0 for null concentrations? (i.e. 0.1) Possibly depending on strenght of connections
    # between healthy and infected regions (and also internal concentrations)
    connect_matrix = connect_matrix / np.max(connect_matrix)
    
    # Define gaussian noise (NOTE authors don't differentiate it by region nor by time)
    noise = np.random.normal(mu_noise, sigma_noise)
    
    iterations = int(years/timestep)
    
    #P = define_initial_P(concentration, connect_matrix, distances, max_concentration)
    P = np.array(np.where(concentration>0, 1, 0), dtype=np.float64)
    Beta = np.zeros(n_regions)
    Delta = np.zeros(n_regions)
    
    for _ in range(iterations):
        with warnings.catch_warnings():
            try:
                Beta = 1 - np.exp(- beta_0 * P)
                Delta = np.exp(- delta_0 * P)
                gini = compute_gini(concentration)
                
                Beta_ext = gini * Beta 
                Beta_int = (1 - gini) * Beta
                
                # Epsilon has to be computed at each time step
                Epsilon = np.zeros(n_regions)
                for i in range(n_regions):
                    for j in range(n_regions):
                        if i != j:
                            # NOTE: I am assuming the delay to go from j to i is null (else Beta_ext(t- Tau(ij)))
                            # Computing the extrinsic infection probability (i!=j)...
                            Epsilon[i] += connect_matrix[j,i] * Beta_ext[j] 
                    # and summing the intrinstic infection rate
                    Epsilon[i] += connect_matrix[i,i] * Beta_int[i] * P[i]
            
                # Updating probabilities
                P = (1 - P) * Epsilon - Delta * P + noise
            except Exception as e:
                logging.error(e)
                return concentration
            
        concentration += (P * (concentration @ connect_matrix)) * timestep
        
    return concentration
        
        
def run_simulation(paths, output_dir, subj, beta_0, delta_0, mu_noise, sigma_noise, queue):
    subject_output_dir = os.path.join(output_dir, subj)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome'])) 
        connect_matrix = np.expm1(connect_matrix)
        connect_matrix += 1e-2
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        return
    
    years = 2
    timestep = 0.0001
    try:
        t1_concentration_pred = Simulation(
            t0_concentration.copy(),           # initial concentration
            connect_matrix,             # connectome
            years,                      # t_total
            timestep,                   # dt
            beta_0,                     # beta_0
            delta_0,                    # delta_0
            mu_noise,                   # mu_noise
            sigma_noise,                # sigma_noise
            1,                          # v
            166,                        # N_regions
            )

        t1_concentration_pred = drop_negative_predictions(t1_concentration_pred)
        if np.isnan(t1_concentration_pred).any() or np.isinf(t1_concentration_pred).any(): raise Exception("Discarding prediction")
    except Exception as e:
        logging.error("Error in simulation")
        logging.error(e)
        return
    
    try:
        rmse = calc_rmse(t1_concentration, t1_concentration_pred)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(rmse) or np.isinf(rmse): raise Exception("Invalid value of RMSE")
        if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(e)
        return
    '''
    visualize_terminal_state_comparison(t0_concentration, 
                                        t1_concentration_pred, 
                                        t1_concentration, 
                                        subj,
                                        rmse,
                                        corr_coef)
    '''
    if queue:
        queue.put([rmse, corr_coef])
        
    return

if __name__=="__main__":
    total_time = time()

    if os.getcwd().endswith('simulations'):
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
    output_dir = 'results'

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
            
    beta_0 = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while num_cores < 0:
        try:
            beta_0 = float(input('Insert the value for beta_0: '))
        except Exception as e:
            logging.error(e)
    
    delta_0 = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while delta_0 < 0:
        try:
            delta_0 = float(input('Insert the value for delta_0: '))
        except Exception as e:
            logging.error(e)
    
    mu_noise = float(sys.argv[5]) if len(sys.argv) > 5 else -1
    while mu_noise < 0:
        try:
            mu_noise = float(input('Insert the value for mu_noise: '))
        except Exception as e:
            logging.error(e)
            
    sigma_noise = float(sys.argv[6]) if len(sys.argv) > 6 else -1
    while sigma_noise < 0:
        try:
            sigma_noise = float(input('Insert the value for sigma_noise: '))
        except Exception as e:
            logging.error(e)

    procs = []
    queue = multiprocessing.Queue()
    total_rmse = []
    total_pcc = []
    
    for subj, paths in tqdm(dataset.items()):
        p = multiprocessing.Process(target=run_simulation, args=(
            paths, 
            output_dir, 
            subj,
            beta_0, 
            delta_0, 
            mu_noise, 
            sigma_noise, 
            queue))
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
        rmse, pcc = queue.get()
        total_rmse.append(rmse)
        total_pcc.append(pcc)
   
    pt_avg.add_row([format(np.mean(total_rmse, axis=0), '.2f'), format(np.std(total_rmse, axis=0), '.2f'), format(np.mean(total_pcc, axis=0), '.2f'), format(np.std(total_pcc, axis=0), '.2f')])

    total_time = time() - total_time
    filename = f"results/{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}_ESM_{category}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Beta_0: {beta_0}\n")
    out_file.write(f"Delta_0: {delta_0}\n")
    out_file.write(f"mu_noise: {mu_noise}\n")
    out_file.write(f"sigma_noise: {sigma_noise}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")