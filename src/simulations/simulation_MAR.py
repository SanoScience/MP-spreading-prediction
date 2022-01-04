''' Spreading model based on Multivariate Autoregressive Model. 

Based on publication: 
A.Crimi et al. "Effective Brain Connectivity Through a Constrained Autoregressive Model" MICCAI 2016
'''

import os
import logging

from tqdm import tqdm 
import numpy as np

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison

logging.basicConfig(level=logging.INFO)

class MARsimulation:
    def __init__(self, connect_matrix, t0_concentrations, t1_concentrations):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.brain_par = 116                                                    # no. of brain areas from the atlas
        self.maxiter = 5000                                                     # max no. of iterations for the gradient descent
        self.th = 0.016                                                         # acceptable error threshold for the reconstruction error
        self.eta = 5e-15                                                        # learning rate of the gradient descent       
        self.cm = connect_matrix
        self.init_concentrations = t0_concentrations
        self.final_concentrations = t1_concentrations

    def run(self, norm_opt, inverse_log=False):
        ''' Run simulation. 
        
        Args:
            norm_opt (int): normalize option for connectivity matrix
            inverse_log (boolean): use normal values instead of logarithmic in connectivity matrix '''
        if inverse_log: self.calc_exponent()
        self.transform_cm(norm_opt)
        self.generate_indicator_matrix()
        coef_matrix = self.run_gradient_descent()
 
    def calc_exponent(self):
        ''' Inverse operation to log1p. '''
        self.cm = np.expm1(self.cm)
        
    def transform_cm(self, norm_opt):
        '''
        Args:
            0 means no normalization, = 1 means binarize, = 2 means divide by the maximum '''
        
        if norm_opt == 0:
            logging.info('No normalization of the initial matrix')
        elif norm_opt == 1:
            logging.info('Initial matrix binarized')
            self.cm = np.where(self.cm > 0, 1, 0).astype('float32')
        elif norm_opt == 2:
            logging.info('Initial matrix normalized according to its largest value')
            max_val = np.max(self.cm)
            self.cm /= max_val
        else:
            logging.info('No normalization of the initial matrix')
            
    def generate_indicator_matrix(self):
        ''' Construct a matrix with only zeros and ones to be used to 
        reinforce the zero connection (this is **B** in our paper).
        noconn_M has zero elements where no structural connectivity appears. '''
        self.noconn_M = np.where(self.cm==0, 0, 1).astype('float32')
        
    def run_gradient_descent(self):
        iter_count = 0  # counter of the current iteration 
        etem = [] # reconstruction error along iterations
        error_des = 1e10 # initial error of reconstruction 
        M = self.cm # initial value for coefficient matrix is connectivity matrix 

        # loop direct connections until criteria are met 
        while (error_des > self.th) and (iter_count < self.maxiter):
            gradient = np.zeros((self.brain_par, self.brain_par))
            # calculate reconstruction error 
            error_des = 0.5 * np.linalg.norm(self.final_concentrations - M @ self.init_concentrations)
            etem.append(error_des)
            # TODO: gradient computation; verify with Alex; grandient values are really high
            gradient += ((self.final_concentrations - M @ self.init_concentrations) * self.init_concentrations)
            # gradient += (M @ self.init_concentrations @ self.init_concentrations.T - self.final_concentrations @ self.init_concentrations.T)
            # update rule
            M -= self.eta * gradient
            # reinforce where there was no connection at the beginning 
            M *= self.noconn_M
            # TODO: remove negative values?
            # M *= (M > 0)
            iter_count += 1
            
        print(etem)
        # print(etem)
        return M
            
    def save_diffusion_matrix(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'diffusion_matrix_over_time.csv'), 
                                self.diffusion_final, delimiter=",")
    
    def save_terminal_concentration(self, save_dir):
        ''' Save last (terminal) concentration. '''
        np.savetxt(os.path.join(save_dir, 'terminal_concentration.csv'),
                   self.diffusion_final[-1, :], delimiter=',')
               
def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data
    
def run_simulation(connectomes_dir, concentrations_dir, output_dir, subject):    
    ''' Run simulation for single patient. '''
      
    connectivity_matrix_path = os.path.join(os.path.join(connectomes_dir, subject), 
                                            'connect_matrix_rough.csv')
    t0_concentration_path = os.path.join(concentrations_dir, subject, 
                                         f'nodeIntensities-not-normalized-{subject}t0.csv')
    t1_concentration_path = os.path.join(concentrations_dir, subject, 
                                         f'nodeIntensities-not-normalized-{subject}t1.csv')
    subject_output_dir = os.path.join(output_dir, subject)
    
    # load connectome
    connect_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brian regions
    t0_concentration = load_matrix(t0_concentration_path) 
    t1_concentration = load_matrix(t1_concentration_path)
            
    simulation = MARsimulation(connect_matrix, t0_concentration, t1_concentration)
    simulation.run(norm_opt=2)
    
def main():
    connectomes_dir = '../../data/connectomes'
    concentrations_dir = '../../data/PET_regions_concentrations'
    output_dir = '../../results' 
    
    patients = ['sub-AD6264'] #['sub-AD4215', 'sub-AD4500', 'sub-AD6264']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(connectomes_dir, concentrations_dir, output_dir, subject)
    
if __name__ == '__main__':
    main()