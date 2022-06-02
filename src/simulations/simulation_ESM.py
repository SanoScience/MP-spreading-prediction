"""
    SYNOPSIS
    python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise> <iterations>
"""

''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

Based on publication: 
"Epidemic Spreading Model to Characterize Misfolded Proteins Propagation 
in Aging and Associated Neurodegenerative Disorders"
Authors: Yasser Iturria-Medina ,Roberto C. Sotero,Paule J. Toussaint,Alan C. Evans 
'''

from datetime import datetime
import json
import multiprocessing 
import os
import logging
import sys
from time import time
import warnings
from prettytable import PrettyTable
import yaml
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_error

from utils_vis import save_prediction_plot
from utils import drop_data_in_connect_matrix, load_matrix

np.seterr(all = 'raise')
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_ESM_{date}.log")

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


def Simulation(concentration, connect_matrix, beta_0, delta_0, mu_noise, sigma_noise, iterations, timestep, velocity=1, n_regions = 166):
    
    #TODO: should I use something different from 0 for null concentrations? (i.e. 0.1) Possibly depending on strenght of connections
    # between healthy and infected regions (and also internal concentrations)
    
    # Define gaussian noise (NOTE authors don't differentiate it by region nor by time)
    noise = np.random.normal(mu_noise, sigma_noise)
    
    #P = define_initial_P(concentration, connect_matrix, distances, max_concentration)
    # NOTE Initial P has to be explored for different values (i.e. 0.5 instead of 1)
    P = np.array(np.where(concentration>0, 1, 0), dtype=np.float64)
    Beta = np.zeros(n_regions)
    Delta = np.zeros(n_regions)
    
    for _ in range(iterations):
        with warnings.catch_warnings():
            warnings.filterwarnings ('error')
            try:
                Beta = 1 - np.exp(- beta_0 * P) # Diffusion
                Delta = np.exp(- delta_0 * P) # Recovery
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
                #P = (1 - P) * Epsilon - Delta * P + noise
                P = (1 - P) * Epsilon - Delta * P + noise
                assert P.all()<=1 and P.all()>=0, "Probabilities are not between 0 and 1"
            except Exception as e:
                logging.error(e)
                return concentration
            
        concentration += (P * (concentration @ connect_matrix)) * timestep
        
    return concentration
        
        
def run_simulation(paths, subj, beta_0, delta_0, mu_noise, sigma_noise, iterations, timestep, queue):      
    try:
        connect_matrix = drop_data_in_connect_matrix(load_matrix(paths['CM']))
        #connect_matrix = prepare_cm(connect_matrix)
        # ESM uses self connections for inner propagation
        connect_matrix += np.diag(connect_matrix)
        t0_concentration = load_matrix(paths['baseline'])
        t1_concentration = load_matrix(paths['followup'])
    except Exception as e:
        logging.error(f'Error appening while loading data of subject {subj}. Traceback: ')
        logging.error(e)
        queue.put([subj, -1, -1])
        return

    try:
        t1_concentration_pred = Simulation(
            
            t0_concentration.copy(),           # initial concentration
            connect_matrix,             # CM                   
            beta_0,                     # beta_0
            delta_0,                    # delta_0
            mu_noise,                   # mu_noise
            sigma_noise,                # sigma_noise
            iterations,
            timestep,
            1,                          # v
            166,                        # N_regions
            )

        t1_concentration_pred = drop_negative_predictions(t1_concentration_pred)
        if np.isnan(t1_concentration_pred).any() or np.isinf(t1_concentration_pred).any(): raise Exception("Discarding prediction")
    except Exception as e:
        logging.error(f'Error during simulation for subject {subj}. Traceback: ')
        logging.error(e)
        queue.put([subj, -1, -1])
        return
    
    try:
        mse = mean_squared_error(t1_concentration, t1_concentration_pred)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if np.isnan(mse) or np.isinf(mse): raise Exception("Invalid value of MSE")
        if np.isnan(corr_coef): raise Exception("Invalid value of PCC")
    except Exception as e:
        logging.error(f'Error appening during computation of MSE and PCC for subject {subj}. Traceback: ')
        logging.error(e)
        queue.put([subj, -1, -1])
        return
    
    save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subj, subj + 'ESM_' + date + '.png', mse, corr_coef)
    logging.info(f"Saving prediction in {subj + 'ESM_' + date + '.png'}")
    if queue:
        queue.put([subj, mse, corr_coef])
        
    return


if __name__=="__main__":
    total_time = time()

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if os.getcwd().endswith('simulations'):
        os.chdir(os.getcwd()+'/../../..')
    category = sys.argv[1] if len(sys.argv) > 1 else ''
    while category == '':
        try:
            category = input('Insert the category [ALL, AD, LMCI, MCI, EMCI, CN; default ALL]: ')
        except Exception as e:
            logging.info("Using default value")
            category = 'ALL'
        category = 'ALL' if category == '' else category

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + 'simulations/'
    if not os.path.exists(output_res):
        os.makedirs(output_res)

    pt_avg = PrettyTable()
    pt_avg.field_names = ["Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
    
    pt_subs = PrettyTable()
    pt_subs.field_names = ["ID", "MSE", "Pearson"]
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
            
    beta_0 = float(sys.argv[3]) if len(sys.argv) > 3 else -1
    while beta_0 < 0:
        try:
            beta_0 = float(input('Insert the value for beta_0 [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            beta_0 = 0
    
    delta_0 = float(sys.argv[4]) if len(sys.argv) > 4 else -1
    while delta_0 < 0:
        try:
            delta_0 = float(input('Insert the value for delta_0 [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            delta_0 = 0
    
    mu_noise = float(sys.argv[5]) if len(sys.argv) > 5 else -1
    while mu_noise < 0:
        try:
            mu_noise = float(input('Insert the value for mu_noise [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            mu_noise = 0
            
    sigma_noise = float(sys.argv[6]) if len(sys.argv) > 6 else -1
    while sigma_noise < 0:
        try:
            sigma_noise = float(input('Insert the value for sigma_noise [default 0]: '))
        except Exception as e:
            logging.info('Using default value')
            sigma_noise = 0
            
    iterations = int(sys.argv[7]) if len(sys.argv) > 7 else -1
    while iterations < 0:
        try:
            iterations = int(input('Insert the number of iterations [default 20\'000]: '))
        except Exception as e:
            logging.info('Using default value')
            iterations = 20000
            
    timestep = float(sys.argv[8]) if len(sys.argv) > 8 else -1
    while timestep < 0 or timestep > 1:
        try:
            timestep = float(input('Insert timestep [default 0.001]: '))
        except Exception as e:
            logging.info("Using default value")
            timestep = 0.001

    logging.info('***********************')
    logging.info(f"Category: {category}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Beta_0: {beta_0}")
    logging.info(f"Delta_0: {delta_0}")
    logging.info(f"mu_noise: {mu_noise}")
    logging.info(f"Iterations: {iterations}")
    logging.info(f"Timestep: {timestep}")
    logging.info(f"sigma_noise: {sigma_noise}")
    logging.info(f"Subjects: {len(dataset.keys())}")
    logging.info('***********************')
    
    procs = []
    queue = multiprocessing.Queue()
    total_mse = []
    total_pcc = []
    
    counter = 0
    done = 0
    for subj, paths in tqdm(dataset.items()):
        p = multiprocessing.Process(target=run_simulation, args=(
            paths, 
            subj,
            beta_0, 
            delta_0, 
            mu_noise, 
            sigma_noise, 
            iterations,
            timestep,
            queue))
        p.start()
        procs.append(p)
        counter+=1

        if counter%num_cores == 0 and counter > 0:
            subj, mse, pcc = queue.get()
            # mse and pcc are -1 if an exception happened during simulation, and are therefore not considered
            if mse != -1 and pcc != -1:
                total_mse.append(mse)
                total_pcc.append(pcc)
            pt_subs.add_row([subj, round(mse,2), round(pcc,2)])
            counter -= 1
            done += 1
    
    while done < len(procs):
        subj, mse, pcc = queue.get()
        if mse != -1 and pcc != -1:
            total_mse.append(mse)
            total_pcc.append(pcc)
        pt_subs.add_row([subj, round(mse,2), round(pcc,2)])
        done += 1
   
    pt_avg.add_row([format(np.mean(total_mse, axis=0), '.2f'), format(np.std(total_mse, axis=0), '.2f'), format(np.mean(total_pcc, axis=0), '.2f'), format(np.std(total_pcc, axis=0), '.2f')])

    total_time = time() - total_time
    filename = f"{output_res}ESM_{category}_{beta_0}_{delta_0}_{mu_noise}_{sigma_noise}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
    out_file = open(filename, 'w')
    out_file.write(f"Category: {category}\n")
    out_file.write(f"Cores: {num_cores}\n")
    out_file.write(f"Beta_0: {beta_0}\n")
    out_file.write(f"Delta_0: {delta_0}\n")
    out_file.write(f"mu_noise: {mu_noise}\n")
    out_file.write(f"sigma_noise: {sigma_noise}\n")
    out_file.write(f"Iterations: {iterations}\n")
    out_file.write(f"Timestep: {timestep}\n")
    out_file.write(f"Subjects: {len(dataset.keys())}\n")
    out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
    out_file.write(pt_avg.get_string()+'\n')
    out_file.write(pt_subs.get_string())    
    out_file.close()