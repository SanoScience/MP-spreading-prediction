''' Utilities funciton for calculation purposes. '''

import numpy as np
from sklearn.metrics import mean_squared_log_error

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def calc_rmse(output, target):
    ''' Compare output from simulation with 
    the target data extracted from PET using MSE metric. '''
    RMSE = np.sqrt(np.sum((output - target)**2) / len(output))
    return RMSE 

def calc_msle(output, target):
    ''' Compare output from simulation with 
    the target data extracted from PET using Mean squared logarithmic error metric. '''
    return mean_squared_log_error(target, output)