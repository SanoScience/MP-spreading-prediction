''' Simulation of spreading the misfolded beta_amyloid with 
Intra-brain Epidemic Spreading model. 

# A. Randriatahina 
# Adapted from Iturria-Medina, Yasser, et al, PloS Biology 2014
'''

from glob import glob 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

def dijkstra(matrix):
    ''' Find the shortest path between nodes in a graph. '''
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

def Simulation(N_regions, v, dt, T_total, Sconnectom , sconnLen, ROI, amy_control, init_number, prob_stay, trans_rate):
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
    

    Sconnectom = Sconnectom - np.diag(np.diag(Sconnectom))
    sconnLen = sconnLen - np.diag(np.diag(sconnLen))

    #set the mobility pattern
    connect = Sconnectom
    beta0 = 1 * trans_rate /ROI 
    g = 1 #global tuning variable that  quantifies the temporal MP deposition inequality 
    #among the different brain regions
    mu_noise = 0 #mean of the additive noise
    sigma_noise = 1 # standard deviation of the additive noise
    Ki = np.random.normal(mu_noise, sigma_noise, (N_regions,N_regions))
    #regional probability receiving MP infectous-like agents
    beta = 1 - np.exp(-beta0 *prob_stay)
    Epsilon = np.zeros((N_regions, N_regions))
    for i in range(N_regions):
        t = 0
        for j in range(N_regions):
            if i != j:
                t =  t +  beta * prob_stay * (Sconnectom[i, j] * g + Sconnectom[i, i] * (1 - g))
        Epsilon[i] = t
    #INTRA-BRAIN EPIDEMIC SPREADING MODEL 
    connect = (1 - prob_stay)* Epsilon - prob_stay * np.exp(- init_number* prob_stay) +  Ki
    
    #The probability of moving from region i to edge (i,j)
    sum_line= [sum(connect[i]) for i in range(N_regions)]
    Total_el_col= np.tile(np.transpose(sum_line), (1,1)) 
    connect = connect / Total_el_col
    
    
    clearance_rate = np.ones(N_regions)
    synthesis_rate = np.ones(N_regions)

    #store the number of misfolded beta-amyloid at each time step
    Rmis_all = np.zeros((N_regions, T_total))
    Pmis_all = np.zeros((N_regions,N_regions, T_total))


    # Rmis, Pmis store results of single simulation at each time
    Rnor = np.zeros(N_regions)# number of  normal amyloid in regions
    Rmis = np.zeros(N_regions)# number of misfolded beta-amyloid in regions
    Pnor = np.zeros(N_regions)# number of  normal amyloid in paths
    Pmis = np.zeros((N_regions, N_regions)) # number of misfolded beta-amyloid in paths



    # fill the network with normal proteins
    iter_max = 50
    epsilon = 1e-2 # We choose this small espilon to avoid division with zero 
    sconnLen = sconnLen + epsilon 

    movOut = np.zeros((N_regions, N_regions))
    #normal amyloid protein growth
    for t in tqdm(range(iter_max)):  
    ##moving process
    # regions towards paths
    # movDrt stores the number of proteins towards each region. i.e. element in kth row lth col denotes the number of proteins in region k moving towards l
        movDrt = Rnor * connect
        movDrt = movDrt * dt 
        movDrt = movDrt - np.diag(np.diag(movDrt))
    
    
        # paths towards regions
        # update moving
        movOut = (v * Pnor)  / sconnLen
        movOut = movOut - np.diag(np.diag(movOut))

        Pnor = Pnor - movOut * dt + movDrt
        Pnor = Pnor-  np.diag(np.diag(Pnor))
        Sum_rows_movOut = [sum(movOut[i])for i in range(N_regions)]
        Sum_cols_movDrt = [sum(movDrt[:,i]) for i in range(N_regions)]

        Rtmp = Rnor
        Rnor = Rnor +  np.transpose(Sum_rows_movOut) * dt -  Sum_cols_movDrt

        #growth process	
        Rnor = Rnor -  Rnor * (1 - np.exp(- clearance_rate * dt))   + (synthesis_rate * amy_control) * dt
        if np.absolute(Rnor - Rtmp).all() < (1e-7 * Rtmp).all():
            break
	
 
    Pnor0 = Pnor
    Rnor0 = Rnor


    # misfolded protein spreading process
    movOut_mis = np.zeros((N_regions, N_regions))
    movDrt_mis = np.zeros((N_regions, N_regions))
    #seed regions
    # Rmis[97] = init_number
    # Rmis[99] = init_number
    Rmis[[31, 32, 35, 36]] = init_number
    for t  in range (T_total):
        #moving process
        # misfolded proteins: region -->> paths
        movDrt_mis= Rnor0 * connect * dt
        movDrt_mis = movDrt_mis - np.diag(np.diag(movDrt_mis))
        
        #normal proteins: paths -->> regions
        movOut_mis = (Pnor0 / sconnLen)*v 
        movOut_mis = movOut_mis - np.diag(np.diag(movOut_mis))
    
        #update regions and paths
        Pmis = Pmis - movOut_mis * dt +  movDrt_mis
        Pmis = Pmis - np.diag(np.diag(Pmis))

        Sum_rows_movOut_mis = [sum(movOut_mis[i]) for i in range(N_regions)]
        Sum_cols_movDrt_mis = [sum(movDrt_mis[:,i]) for i in range(N_regions)]

        Rmis = Rmis + np.transpose(Sum_rows_movOut_mis ) * dt  - Sum_cols_movDrt_mis


        Rmis_cleared = Rmis * (1 - np.exp(-clearance_rate * dt))
  
        #the probability of getting misfolded
        misProb = 1 - np.exp( - Rmis * beta0 * dt ) 
        #number of newly infected
        N_misfolded = Rnor0 * np.exp(- clearance_rate) * misProb 
  
        #update
        Rmis = Rmis - Rmis_cleared + N_misfolded + (synthesis_rate * amy_control) *dt
        
        #Depostion of misfolded protein
        #Regions
        Rmis_all[:,t] = Rmis

        #paths
        Pmis_all[:, :, t] = Pmis

    return  Rmis_all, Pmis_all, connect

def surface_plot2d(matrix):
    ''' Plot heatmap. '''
    plt.imshow(matrix, cmap = 'jet');
    plt.colorbar()
    t = plt.show()
    return t

def main():
    N_regions = 116  #= 184 
    v = 1
    dt = 0.01 
    T_total = 50
    ROI = 184 # or 116?
    amy_control = 3
    init_number = 0.2
    prob_stay = 0.5
    trans_rate = 4
    P = 0
    paths = glob('../data/output/*/connect_matrix_filtered.csv')
    for path in paths:
        subject = np.genfromtxt(path, delimiter=",")
        Sconnectom = subject
        sconnLen = dijkstra(subject)
        Rmis_all, Pmis_all,  weights = Simulation(N_regions, v, dt, T_total, Sconnectom , sconnLen, ROI, amy_control, init_number, prob_stay, trans_rate)
        P = P + 1
        # RESULTS
        print('Sketch for Subject',P)
        print('the connectivity matrix')
        surface_plot2d(subject)
        print('the number of misfolded beta-amyloid in regions')
        print(Rmis_all, np.max(Rmis_all), np.min(Rmis_all))
        surface_plot2d(Rmis_all)
        for t in range(T_total):
            Y = Pmis_all[:, :, t]
        print('a number of misfolded beta_amyloid one path could be memory-consuming')
        surface_plot2d(Y)
        plt.show()

if __name__=="__main__":
    main()
