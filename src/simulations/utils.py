''' Utilities funciton for calculation purposes. '''

import os
from glob import glob

import numpy as np
from sklearn.metrics import mean_squared_log_error

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def get_file_paths_for_subject(dataset_dir, subject, tracer='av45'):
    connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_rough.csv'))
    
    t0_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'pet', f'*baseline*trc-{tracer}_pet.csv')))[0]
    # extract baseline year 
    t0_year = int(t0_concentration_path.split('date-')[1][:4])
    # followup year should be: baseline year + time interval
    time_interval = 2
    t1_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                                           'ses-followup', 'pet', 
                                                           f'*followup*{t0_year+time_interval}*trc-{tracer}_pet.csv')))[0]
    return connectivity_matrix_path, t0_concentration_path, t1_concentration_path

def drop_data_in_connect_matrix(connect_matrix, missing_labels=[35, 36, 81, 82]):
    index_to_remove = [(label - 1) for label in missing_labels]
    connect_matrix = np.delete(connect_matrix, index_to_remove, axis=0)
    connect_matrix = np.delete(connect_matrix, index_to_remove, axis=1) 
    return connect_matrix

def save_diffusion_matrix(save_dir, diffusion_matrix, method_name):
    np.savetxt(os.path.join(save_dir, f'diffusion_matrix_over_time_{method_name}.csv'), 
                            diffusion_matrix, delimiter=",")
    
def save_coeff_matrix(save_dir, matrix):
    np.savetxt(os.path.join(save_dir, f'A_matrix_MAR.csv'), 
                            matrix, delimiter=",")
    
def save_terminal_concentration(save_dir, concentration_pred, method_name):
    ''' Save last (terminal) concentration. '''
    np.savetxt(os.path.join(save_dir, f'concentration_pred_{method_name}.csv'),
                concentration_pred, delimiter=',')
    
def drop_negative_predictions(predictions):
    return np.maximum(predictions, 0)

def calc_rmse(output, target):
    ''' Compare output from simulation with 
    the target data extracted from PET using MSE metric. '''
    RMSE = np.sqrt(np.sum((output - target)**2) / len(output))
    return RMSE 

def calc_msle(output, target):
    ''' Compare output from simulation with 
    the target data extracted from PET using Mean squared logarithmic error metric. '''
    return mean_squared_log_error(target, output)