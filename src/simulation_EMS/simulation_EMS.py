''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

Author: A. Randriatahina 

Based on publication: 
"Epidemic Spreading Model to Characterize Misfolded Proteins Propagation 
in Aging and Associated Neurodegenerative Disorders"
Authors: Yasser Iturria-Medina ,Roberto C. Sotero,Paule J. Toussaint,Alan C. Evans 
'''

# TODO: refactor this code 
# TODO: save prediction and plots into /results/ directory 


from glob import glob 
import os
import logging
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.shortest_paths.weighted import _dijkstra

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

class EMS_Simulation:
    # A class to simulate the spread of misfolded beta_amyloid

    #GBA: GBA gene expression (zscore, N_regions * 1 vector) (empirical GBA expression)
    #SNCA: SNCA gene expression after normalization (zscore, N_regions * 1 vector) (empirical SNCA expression)

    def __init__(self, connect_matrix_in, concentrations, sconnLens, years):
        # constants used in calculation
        self.N_regions = 116    # N_regions: number of regions 
        self.v = 1              # v: speed (1)
        self.dt = 0.01          # dt: time step (0.01)
        self.T_total = years    # total time steps (10000)
        self.ROI = 116          # TODO documentation
        self.amy_control = 3    # TODO documentation
        self.prob_stay = 0.5    # the probability of staying in the same region per unit time (0.5)
        self.trans_rate = 4     # a scalar value, controlling the baseline infectivity
        self.P = 0              # TODO documentation
        self.iter_max = 50      # fill the network with normal proteins
        self.beta0 = 1 * self.trans_rate / self.ROI 
        self.mu_noise = 0       # mean of the additive noise
        self.sigma_noise = 1    # standard deviation of the additive noise
        self.diffusion_init = concentrations
        self.mu_noise = 0       # mean of the additive noise
        self.sigma_noise = 1    # standard deviation of the additive noise
        self.connect_matrix = connect_matrix_in - np.diag(np.diag(connect_matrix_in)) # connectivity matrix of the patient
        self.clearance_rate = np.ones(self.N_regions)   # TODO documentation
        self.synthesis_rate = np.ones(self.N_regions)   # TODO documentation
        
        epsilon = 1e-2 # We choose this small espilon to avoid division with zero 
        sconnLen1 = sconnLens - np.diag(np.diag(sconnLens))    # sconnLen: structural connectivity matrix (length) (estimated from HCP data)
        self.sconnLen = sconnLen1 + epsilon 

    def calculate_connect(self):
        g = 1 #global tuning variable that quantifies the temporal MP deposition inequality 

        Ki = np.random.normal(self.mu_noise, self.sigma_noise, (self.N_regions,self.N_regions))
        
        # regional probability receiving MP infectous-like agents
        beta = 1 - np.exp(-self.beta0 *self.prob_stay)
        Epsilon0 = np.zeros((self.N_regions, self.N_regions))
        
        for i in range(self.N_regions):
            t = 0
            for j in range(self.N_regions):
                if i != j:
                    t =  t +  beta * self.prob_stay * (self.connect_matrix[i, j] * g + self.connect_matrix[i, i] * (1 - g))
            Epsilon0[i] = t
        # INTRA-BRAIN EPIDEMIC SPREADING MODEL 
        connect = (1 - self.prob_stay)* Epsilon0 - self.prob_stay * np.exp(- self.diffusion_init* self.prob_stay) +  Ki
        
        # The probability of moving from region i to edge (i,j)
        sum_line= [sum(connect[i]) for i in range(self.N_regions)]
        Total_el_col= np.tile(np.transpose(sum_line), (1,1)) 
        connect = connect / Total_el_col

        return connect
    
    def calculate_Rnor_Pnor(self, connect):
        
        Rnor = np.zeros(self.N_regions) # number of  normal amyloid in regions
        Pnor = np.zeros(self.N_regions) # number of  normal amyloid in paths

        movOut = np.zeros((self.N_regions, self.N_regions))

        #normal amyloid protein growth
        for t in tqdm(range(self.iter_max)):  
        ## moving process
        # regions towards paths
        # movDrt stores the number of proteins towards each region. i.e. element in kth row lth col denotes the number of proteins in region k moving towards l
            movDrt = Rnor * connect
            movDrt = movDrt * self.dt 
            movDrt = movDrt - np.diag(np.diag(movDrt))
        
            # paths towards regions
            # update moving
            movOut = (self.v * Pnor)  / self.sconnLen
            movOut = movOut - np.diag(np.diag(movOut))

            Pnor = Pnor - movOut * self.dt + movDrt
            Pnor = Pnor-  np.diag(np.diag(Pnor))
            Sum_rows_movOut = [sum(movOut[i])for i in range(self.N_regions)]
            Sum_cols_movDrt = [sum(movDrt[:,i]) for i in range(self.N_regions)]

            Rtmp = Rnor
            Rnor = Rnor +  np.transpose(Sum_rows_movOut) * self.dt -  Sum_cols_movDrt

            #growth process	
            Rnor = Rnor -  Rnor * (1 - np.exp(- self.clearance_rate * self.dt))   + (self.synthesis_rate * self.amy_control) * self.dt
            if np.absolute(Rnor - Rtmp).all() < (1e-7 * Rtmp).all():
                break
    
        return Rnor, Pnor

    def calculate_Rmis_Pmis(self, connect, Rnor0, Pnor0):
        # Pmis_all: a N_regions * N_regions * T_total matrix, recording the number of misfolded beta_amyloid in paths could be memoryconsuming
        # Rmis_all: A N_regions * T_total matrix, recording the number of misfolded beta-amyloid in regions
        # Rnor0: a N_Regions * 1 vector, the population of normal agents in regions before pathogenic spreading 
        # Pnor0: a N_Regions * 1 vecotr, the population of normal agents in edges before pathogenic spreading 

        # Rmis, Pmis store results of single simulation at each time
        Rmis = np.zeros(self.N_regions)                         # number of misfolded beta-amyloid in regions
        Pmis = np.zeros((self.N_regions, self.N_regions))       # number of misfolded beta-amyloid in paths

        # store the number of misfolded beta-amyloid at each time step
        Rmis_all = np.zeros((self.N_regions, self.T_total))
        Pmis_all = np.zeros((self.N_regions, self.N_regions, self.T_total))
        
        # misfolded protein spreading process
        movOut_mis = np.zeros((self.N_regions, self.N_regions))
        movDrt_mis = np.zeros((self.N_regions, self.N_regions))

        Rmis = self.diffusion_init
        for t  in range (self.T_total):
            # moving process
            # misfolded proteins: region -->> paths
            movDrt_mis= Rnor0 * connect * self.dt
            movDrt_mis = movDrt_mis - np.diag(np.diag(movDrt_mis))
            
            # normal proteins: paths -->> regions
            movOut_mis = (Pnor0 / self.sconnLen)* self.v 
            movOut_mis = movOut_mis - np.diag(np.diag(movOut_mis))
        
            # update regions and paths
            Pmis = Pmis - movOut_mis * self.dt +  movDrt_mis
            Pmis = Pmis - np.diag(np.diag(Pmis))

            Sum_rows_movOut_mis = [sum(movOut_mis[i]) for i in range(self.N_regions)]
            Sum_cols_movDrt_mis = [sum(movDrt_mis[:,i]) for i in range(self.N_regions)]

            Rmis = Rmis + np.transpose(Sum_rows_movOut_mis ) * self.dt  - Sum_cols_movDrt_mis
            Rmis_cleared = Rmis * (1 - np.exp(-self.clearance_rate * self.dt))
    
            # he probability of getting misfolded
            misProb = 1 - np.exp( - Rmis * self.beta0 * self.dt)

            # number of newly infected
            N_misfolded = Rnor0 * np.exp(- self.clearance_rate) * misProb 
    
            # update
            Rmis = Rmis - Rmis_cleared + N_misfolded + (self.synthesis_rate * self.amy_control) * self.dt
            
            # Depostion of misfolded protein
            # Regions
            Rmis_all[:,t] = Rmis
            # paths
            Pmis_all[:, :, t] = Pmis   
            
        return Rmis_all, Pmis_all
    
    def surface_plot2d(self, matrix):
        ''' Plot heatmap. '''
        plt.imshow(matrix, cmap = 'jet')
        plt.colorbar()
        return plt.show()

    def plot_predicted_vs_real(self, predicted, real, year, error):
        x_axis = np.arange(116)
        plt.plot(x_axis, predicted, label='predicted', c = 'r')
        plt.plot(x_axis, real, label='real', c = 'g')
        plt.grid()
        plt.xlabel('ROI')
        plt.ylabel('concentrations')
        plt.title(f'After {year} years, error: {error:.2f}')
        plt.legend()
        plt.show()
        
    def calc_error(self, output, target):
        ''' Compare output from simulation with 
        the target data extracted from PET using MSE metric. '''
        RMSE = np.sqrt(np.sum((output - target)**2) / len(output))
        return RMSE 
    

def run_simulation(connectomes_dir, concentrations_dir, output_dir, subject):    
    connectivity_matrix_path = os.path.join(os.path.join(connectomes_dir, subject), 
                                        'connect_matrix_rough.csv')
    t0_concentration_path = os.path.join(concentrations_dir, subject, 
                                        f'nodeIntensities-not-normalized-{subject}t0.csv')
    t1_concentration_path = os.path.join(concentrations_dir, subject, 
                                        f'nodeIntensities-not-normalized-{subject}t1.csv')
    subject_output_dir = os.path.join(output_dir, subject)
    years = 2

    # load connectome
    connectivity_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brian regions
    concentrations_init = load_matrix(t0_concentration_path)
    real_concentration = load_matrix(t1_concentration_path)
    
    # the simulation
    shortest = dijkstra(connectivity_matrix)
    simulation = EMS_Simulation(connectivity_matrix, concentrations_init, shortest, years)
    connect = simulation.calculate_connect()
    Rnor, Pnor = simulation.calculate_Rnor_Pnor(connect)
    Rmis_all, Pmis_all = simulation.calculate_Rmis_Pmis(connect, Rnor, Pnor)

    # RESULTS
    print(f'Sketch for Subject: {subject}')

    print('the number of misfolded beta-amyloid in regions')

    # resulting concentration matrix
    print(Rmis_all, np.max(Rmis_all), np.min(Rmis_all))
    simulation.surface_plot2d(Rmis_all)

    # predicted vs real plot
    predicted = Rmis_all[:, years-1]
    simulation.plot_predicted_vs_real(predicted, real_concentration, years, simulation.calc_error(real_concentration, predicted))

    for t in range(simulation.T_total):
        Y = Pmis_all[:, :, t]
    
    print('a number of misfolded beta_amyloid one path could be memory-consuming')
    simulation.surface_plot2d(Y)
    plt.show()

def dijkstra(matrix):
        # Find the shortest path between nodes in a graph.
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

def load_matrix(path):
    return np.genfromtxt(path, delimiter=",")

def main():
    connectomes_dir = '../../data/connectomes'
    concentrations_dir = '../../data/PET_regions_concentrations'
    output_dir = '../../results' 
    
    patients = ['sub-AD6264'] #['sub-AD4215', 'sub-AD4500', 'sub-AD6264']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(connectomes_dir, concentrations_dir, output_dir, subject)
        
        
if __name__=="__main__":
    main()
