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

from scipy.stats import norm
from scipy.stats import zscore

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import drop_data_in_connect_matrix, load_matrix, calc_rmse, calc_msle, save_terminal_concentration

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.DEBUG)

def static_connectivity_matrix(subject):
  corr= np.corrcoef(subject)
  return corr


def Simulation(N_regions, v, dt, T_total, GBA, SNCA, sconnLen, sconnDen, ROIsize, seed, syn_control, init_number, prob_stay, trans_rate):
	
    # A function to simulate the spread of misfolded beta_amyloid

    ##input parameters (inside parenthesis are values used in the paper)	
    #N_regions: number of regions 
    #v: speed (1)
    # dt: time step (0.01)
    # T_total: total time steps (10000)
    #GBA: GBA gene expression (zscore, N_regions * 1 vector) (empirical GBA expression)
    #SNCA: SNCA gene expression after normalization (zscore, N_regions * 1 vector) (empirical SNCA expression)
    # sconnLen: structural connectivity matrix (length) (estimated from HCP data)
    # sconnDen: structural connectivity matrix (strength) (estimated from HCP data)
    # ROIsize: region sizes (voxel counts)
    # seed: seed region of misfolded beta-amyloid injection (choose as you like? )
    # syn_control: a parameter to control the number of voxels in which beta-amyloid may get synthesized (region size, i.e., ROIsize)
    # init_number: number of injected misfolded beta-amyloid (1) 
    # prob_stay: the probability of staying in the same region per unit time (0.5)
    # trans_rate: a scalar value, controlling the baseline infectivity

    ## output parameters
    # Rnor_all: A N_regions * T_total matrix, recording the number of normal beta-amyloid in regions
    # Rmis_all: A N_regions * T_total matrix, recording the number of misfolded beta-amyloid in regions
    # Rnor0: a N_Regions * 1 vector, the population of normal agents in regions before pathogenic spreading 
    # Pnor0: a N_Regions * 1 vecotr, the population of normal agents in edges before pathogenic spreading 
    #Pnor_all: a N_regions * N_regions * T_total matrix, recording the number of normal beta_amyloid in paths could be memory-consuming
    #Pmis_all: a N_regions * N_regions * T_total matrix, recording the number of misfolded beta_amyloid in paths could be memoryconsuming

    # TODO: from here step-by-step
    sconnDen = sconnDen - np.diag(np.diag(sconnDen))
    sconnLen = sconnLen - np.diag(np.diag(sconnLen))

    #set the mobility pattern
    weights = sconnDen
    delta0 = 1 * trans_rate /ROIsize 
    g = 0.5 #global tuning variable that  quantifies the temporal MP deposition inequality among the different brain regions
    mu_noise = 0.2 #mean of the additive noise
    sigma_noise = 0.1 # standard deviation of the additive noise
    Ki = np.random.normal(mu_noise, sigma_noise, (N_regions,N_regions))

    #model
    #regional probability receiving MP infectous-like agents

    Epsilon = np.zeros((N_regions, N_regions))
    for i in range(N_regions):
        t = 0
        for j in range(N_regions):
            if i != j:
                t = t +  (sconnDen[i][j] * (g*(1 - np.exp(-delta0 *prob_stay))) * prob_stay + sconnDen[i][i] * (1 - g)*(1 - np.exp( - delta0*prob_stay))* prob_stay)
                Epsilon[i][j] = t
    weights = (1 - prob_stay)* Epsilon - prob_stay * np.exp(- init_number* prob_stay) +  Ki



    #The probability of moving from region i to edge (i,j)
    sum_line= [sum(weights[i]) for i in range(len(weights))]
    Total_el_col= np.tile(np.transpose(sum_line), (1,1))
    weights = weights / Total_el_col 

    #convert gene expression scores to probabilities
    clearance_rate = norm.cdf(zscore(GBA)) 
    synthesis_rate = norm.cdf(zscore(SNCA))

    #store the number of normal/misfolded beta-amyloid at each time step
    Rnor_all = np.zeros((N_regions, T_total))
    Rmis_all = np.zeros((N_regions, T_total))
    Pnor_all = np.zeros((N_regions, T_total))
    Pmis_all = np.zeros((N_regions, T_total))

    #Rnor, Rmis, Pnor, Pmis store results of single simulation at each time
    Rnor = np.zeros(N_regions)# number of normal beta-amyloid in regions
    Rmis = np.zeros(N_regions) #number of misfolded beta-amyloid in regions
    Pnor = np.zeros((N_regions, N_regions)) # number of normal beta-amyloid in paths
    Pmis= np.zeros((N_regions, N_regions)) # number of misfolded beta-amyloid in paths

    ##normal alpha-syn growth 
    # fill the network with normal proteins

    iter_max = 1

    #normal alpha synuclein growth
    for t in range(iter_max):  
        ###moving process
        # regions towards paths
        # movDrt stores the number of proteins towards each region. i.e. moving towards l
        #movDrt = np.kron(np.ones((1, N_regions)), Rnor) * weights
        movDrt = Rnor * weights
        movDrt = movDrt * dt 
        movDrt = movDrt - np.diag(np.diag(movDrt))


        # paths towards regions
        # update moving
            

        movOut = Pnor * v / sconnLen  #longer path & smaller v = lower probability of moving out of paths
        movOut = movOut - np.diag(np.diag(movOut))

        Pnor = Pnor - movOut * dt + movDrt
        Pnor = Pnor -  np.diag(np.diag(Pnor))
        Sum_rows_movOut = [sum(movOut[i])for i in range(len(movOut))]
        Sum_cols_movDrt = [sum(movDrt[:,i]) for i in range(len(movDrt))]

        Rtmp = Rnor
        Rnor = Rnor +  np.transpose(Sum_rows_movOut) * dt -  Sum_cols_movDrt

        #growth process	
        Rnor = Rnor -  Rnor * (1 - np.exp(- clearance_rate * dt))   + (synthesis_rate * syn_control) * dt

    Pnor0 = Pnor
    Rnor0 = Rnor

    # misfolded protein spreading process

    #inject misfolded beat_amyloid
    Rmis[seed] = init_number;

    for t  in range (T_total):
        #moving process
        # normal proteins: region -->> paths
        #movDrt_nor = np.kron(np.ones((1, N_regions)), Rnor) * weights * dt
        movDrt_nor = Rnor * weights * dt
        movDrt_nor = movDrt_nor - np.diag(np.diag(movDrt_nor))     

        movOut_nor = Pnor * v  / sconnLen 
        movOut_nor = movOut_nor - np.diag(np.diag(movOut_nor))


        #misfolded proteins: region -->> paths
        movDrt_mis =  Rnor * weights * dt
        #movDrt_mis = np.kron(np.ones((1, N_regions)), Rnor) * weights * dt
        movDrt_mis = movDrt_mis - np.diag(np.diag(movDrt_mis))
            
        #misfolded proteins: paths -->> regions
        movOut_mis = Pmis * v / sconnLen
        movOut_mis = movOut_mis - np.diag(np.diag(movOut_mis))

        #update regions and paths
        Pnor = Pnor - movOut_nor * dt + movDrt_nor 
        Pnor = Pnor - np.diag(np.diag(Pnor))

        Sum_rows_movOut_nor = [sum(movOut_nor[i]) for i in range(len(movOut_nor))]
        Sum_cols_movDrt_nor = [sum(movDrt_nor[:,i]) for i in range(len(movDrt_nor))]

        Rnor = Rnor + np.transpose(Sum_rows_movOut_nor ) * dt -  Sum_cols_movDrt_nor

        Pmis = Pmis - movOut_mis*dt + movDrt_mis; 
        Pmis = Pmis - np.diag(np.diag(Pmis))
            
        Sum_rows_movOut_mis = [sum(movOut_mis[i]) for i in range(len(movOut_mis))]
        Sum_cols_movDrt_mis = [sum(movDrt_mis[:,i]) for i in range(len(movDrt_mis))]

        Rmis = Rmis + np.transpose (Sum_rows_movOut_mis)*dt - Sum_cols_movDrt_mis    
            
        Rnor_cleared = Rnor * (1 - np.exp(-clearance_rate * dt))
        Rmis_cleared = Rmis * (1 - np.exp(-clearance_rate * dt))
        #the probability of getting misfolded
        delta0 = 1 * trans_rate /ROIsize 
        misProb = 1 - np.exp( - Rmis * delta0 * dt ) # trans_rate: default
        #number of newly infected
        N_misfolded = Rnor * np.exp(- clearance_rate) * misProb 
        #update
        Rnor = Rnor - Rnor_cleared - N_misfolded + (synthesis_rate * syn_control) *dt
        Rmis = Rmis - Rmis_cleared + N_misfolded

        Rnor_all[: , t]= Rnor 
        Rmis_all[: , t] = Rmis 

        #uncomment the following lines if you want outputs of alpha-syn in
        #paths
        #Pnor_ave(:, :, t) = Pnor
        #Pmis_ave(:, :, t) = Pmis
        #additive noise


    return Rnor_all, Rmis_all, Pnor_all, Pmis_all, Rnor0, Pnor0

def run_simulation(paths, output_dir, subject, queue = None):  
    subject_output_dir = os.path.join(output_dir, subject)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['connectome'])) + 1e-2
        static_CM = np.corrcoef(connect_matrix) + 1e-2
        GBA = np.diag(static_CM)
        SNCA = GBA / np.linalg.norm(GBA)
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(e)
        return

    years = 2
    timestep = 0.002
    try:
        Rnor_all, Rmis_all, Pnor_all, Pmis_all, Rnor0, Pnor0 = Simulation(
                                                        166,  # N_regions
                                                        1,                      # v
                                                        0.001,                   # dt
                                                        1,                      # t_total
                                                        GBA,                    # GBA
                                                        SNCA,                   # SNCA
                                                        connect_matrix,         # sconnLen 
                                                        connect_matrix,         # sconnDen
                                                        4,                      # ROIsize
                                                        2,                      # seed
                                                        1,                      # syn_control
                                                        1,                      # init_number
                                                        0.5,                    # prob_stay
                                                        2                       # trans_rate
                                                        )

        t1_concentration_pred = Rmis_all[:, -1]
        if np.isnan(t1_concentration_pred).any() or np.isinf(t1_concentration_pred).any(): raise Exception("Discarding prediction")
    except Exception as e:
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
    else:
    
        visualize_terminal_state_comparison(t0_concentration, 
                                            t1_concentration_pred, 
                                            t1_concentration, 
                                            subject,
                                            rmse,
                                            corr_coef)
    if queue:
        queue.put([rmse, corr_coef])

    '''
    beta_step = 0.0001
    opt_beta0 = opt_rmse = opt_pcc = -1
    for _ in range(beta_iter):
        simulation = ESMSimulation(connect_matrix, regions_distances, years, timestep, concentrations=t0_concentration, beta0=beta0)
        beta0 += beta_step
        t1_concentration_pred = simulation.run()
        #connect = simulation.calculate_connect()
        #Rnor, Pnor = simulation.calculate_Rnor_Pnor(connect)
        #Rmis_all, Pmis_all = simulation.calculate_Rmis_Pmis(connect, Rnor, Pnor)

        #visualize_diffusion_timeplot(Rmis_all, simulation.dt, simulation.t_total)

        # predicted vs real plot; take results from the last step
        #t1_concentration_pred = Rmis_all[:, years-1]
        try:
            rmse = calc_rmse(t1_concentration, t1_concentration_pred)
            corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
            if np.isnan(rmse) or np.isinf(rmse): raise Exception("Invalid value of RMSE")
            if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
        except Exception as e:
            logging.error(e)
            continue
        else:
            if opt_rmse == -1 or rmse < opt_rmse:
                opt_prediction = t1_concentration_pred
                opt_rmse = rmse 
                opt_pcc = corr_coef
                opt_beta0 = beta0
        
        visualize_terminal_state_comparison(t0_concentration, 
                                            opt_prediction, 
                                            t1_concentration, 
                                            subject,
                                            rmse,
                                            corr_coef)
        
        #save_terminal_concentration(subject_output_dir, t1_concentration_pred, 'EMS')
    
    if queue:
        queue.put([opt_beta0, opt_rmse, opt_pcc])
    '''
    
    return
    


if __name__=="__main__":
    total_time = time()

    #os.chdir(os.getcwd()+'/../../')
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

    procs = []
    queue = multiprocessing.Queue()
    total_rmse = []
    total_pcc = []
    
    for subj, paths in dataset.items():
        p = multiprocessing.Process(target=run_simulation, args=(
            paths, 
            output_dir, 
            subj, 
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
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string())
    out_file.close()
    logging.info(f"Results saved in {filename}")