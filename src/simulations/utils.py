''' Utilities funciton for calculation purposes. '''

import numpy as np

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def calc_rmse(output, target):
    ''' Compare output from simulation with 
    the target data extracted from PET using MSE metric. '''
    RMSE = np.sqrt(np.sum((output - target)**2) / len(output))
    return RMSE 