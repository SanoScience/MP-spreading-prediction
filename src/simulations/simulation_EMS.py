''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

Based on publication: 
"Epidemic Spreading Model to Characterize Misfolded Proteins Propagation 
in Aging and Associated Neurodegenerative Disorders"
Authors: Yasser Iturria-Medina ,Roberto C. Sotero,Paule J. Toussaint,Alan C. Evans 
'''

from glob import glob 
import os
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import _dijkstra
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import load_matrix, calc_rmse, calc_msle, save_terminal_concentration

logging.basicConfig(level=logging.INFO)

class EMS_Simulation:
    ''' A class to simulate the spread of misfolded beta_amyloid. '''

    def __init__(self, connect_matrix, regions_distances, years, concentrations=None):
        self.N_regions = 116    # number of brain regions 
        self.speed = 1          # propagation velocity
        self.dt = 0.01          # time step (0.01)
        self.T_total = years    # total time 
        self.amy_control = 3    # parameter to control the number of voxels in which beta-amyloid may get synthesized (?)
        self.prob_stay = 0.5    # the probability of staying in the same region per unit time (0.5)
        self.trans_rate = 4     # a scalar value, controlling the baseline infectivity
        self.iter_max = 50      # max no. of iterations
        self.mu_noise = 0       # mean of the additive noise
        self.sigma_noise = 1    # standard deviation of the additive noise
        self.gini_coeff = 1     # measure of statistical dispersion in a given system, with value 0 reflecting perfect equality and value 1 corresponding to a complete inequality
        self.clearance_rate = np.ones(self.N_regions)   
        self.synthesis_rate = np.ones(self.N_regions)  
        self.beta0 = 1 * self.trans_rate / self.N_regions
        
        if concentrations is not None: 
            logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
        
        self.regions_distances = regions_distances
        self.connect_matrix = connect_matrix
        
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
        beta = 1 - np.exp(self.beta0 * self.prob_stay)  # regional probability receiving MP infectous-like agents
        Epsilon = np.zeros((self.N_regions, self.N_regions))
        
        for i in range(self.N_regions):
            t = 0
            for j in range(self.N_regions):
                if i != j:
                    t +=  beta * self.prob_stay * (self.connect_matrix[i, j] * self.gini_coeff + self.connect_matrix[i, i] * (1 - self.gini_coeff))
            Epsilon[i] = t
            
        # INTRA-BRAIN EPIDEMIC SPREADING MODEL 
        connect = (1 - self.prob_stay)*Epsilon - self.prob_stay * np.exp(- self.diffusion_init* self.prob_stay) +  noise
        
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
        for t in tqdm(range(self.iter_max)):  
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
            precision = 1e-7
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
    
def run_simulation(connectomes_dir, concentrations_dir, output_dir, subject):    
    connectivity_matrix_path = os.path.join(os.path.join(connectomes_dir, subject), 
                                            'connect_matrix_rough.csv')
    concentrations_paths = glob(os.path.join(concentrations_dir, subject, '*.csv'))
    t0_concentration_path = [path for path in concentrations_paths if 'baseline' in path][0]
    t1_concentration_path = [path for path in concentrations_paths if 'followup' in path][0]
    subject_output_dir = os.path.join(output_dir, subject)
    years = 50

    # load connectome
    connectivity_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brain regions
    t0_concentration = load_matrix(t0_concentration_path)
    t1_concentration = load_matrix(t1_concentration_path)
    
    logging.info(f'Sum of t0 concentration: {np.sum(t0_concentration)}')
    logging.info(f'Sum of t1 concentration: {np.sum(t1_concentration)}')
    
    regions_distances = dijkstra(connectivity_matrix)
        
    simulation = EMS_Simulation(connectivity_matrix, regions_distances, years, concentrations=t0_concentration)
    connect = simulation.calculate_connect()
    Rnor, Pnor = simulation.calculate_Rnor_Pnor(connect)
    Rmis_all, Pmis_all = simulation.calculate_Rmis_Pmis(connect, Rnor, Pnor)

    visualize_diffusion_timeplot(Rmis_all, simulation.dt, simulation.T_total)

    # predicted vs real plot; take results from the last step
    t1_concentration_pred = Rmis_all[:, years-1]
    rmse = calc_rmse(t1_concentration, t1_concentration_pred)
    corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
    visualize_terminal_state_comparison(t0_concentration, 
                                        t1_concentration_pred, 
                                        t1_concentration, 
                                        subject,
                                        rmse,
                                        corr_coef)
    save_terminal_concentration(subject_output_dir, t1_concentration_pred, 'EMS')

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

def main():
    connectomes_dir = '../../data/connectomes'
    concentrations_dir = '../../data/PET_regions_concentrations'
    output_dir = '../../results' 
    
    patients = ['sub-AD4215', 'sub-AD4009']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(connectomes_dir, concentrations_dir, output_dir, subject)
        
        
if __name__=="__main__":
    main()
