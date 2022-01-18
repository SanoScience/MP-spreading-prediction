''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import os
import logging
from glob import glob
from turtle import shape

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import load_matrix, calc_rmse, calc_msle

import networkx as nx

logging.basicConfig(level=logging.INFO)

class DiffusionSimulation:
    def __init__(self, connect_matrix, beta, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        self.beta = beta # As in the Raj et al. papers
        self.rois = 116 # AAL atlas has 116 rois
        self.t_total = 100 # total length of the simulation in years
        self.timestep = 1 # equivalent to 7.3 days per time step
        self.iterations = int(self.t_total / self.timestep) # 200 iterations
        self.cm = connect_matrix
        if concentrations is not None: 
            #logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            #logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
                    
    def run(self, inverse_log=True, downsample=False):
        ''' Run simulation. '''
        if inverse_log: self.calc_exponent()
        self.calc_laplacian()
        
        self.diffusion_final = self.iterate_spreading_by_Julien()
        #self.diffusion_final = self.iterate_spreading()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        return self.diffusion_final[-1]
        
    def define_seeds(self, init_concentration=1):
        ''' Define Alzheimer seed regions manually. 
        
        Args:
            init_concentration (int): initial concentration of misfolded proteins in the seeds. '''
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (Cingulate gyrus, anterior/posterior part, left & right)
        # assign initial concentration of proteins in this region (please note index starts from 0, not from 1, then region 31 is at 30th index in the diffusion array)
        diffusion_init[[30, 31, 34, 35]] = init_concentration
        return diffusion_init
        
    def calc_laplacian(self, eps=1e-10): 
        # Laplacian: L = D - A
        # assume: A - adjacency matrix, D - degree matrix, I - identity matrix, L - laplacian matrix
        self.cm = np.asmatrix(self.cm)
        G = nx.from_numpy_matrix(self.cm)
        # normalized Laplacian: L = I - D-1/2 @ A @ D-1/2
        self.L = nx.normalized_laplacian_matrix(G).toarray()
        
        # this is not the degree matrix
        #D = np.diag(np.sum(A, axis=1))# total no. of. connections to other vertices
        #I = np.identity(A.shape[0]) # identity matrix
        #D_inv_sqrt = np.linalg.inv(np.sqrt(D)+eps) # add epsilon to avoid getting 0 determinant
        #self.L = I - (D_inv_sqrt @ A) @ D_inv_sqrt           

        # eigendecomposition
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        
    def integration_step(self, x0, t):
        # persistent mode of propagation
        # x(t) = U exp(-lambda * beta * t) U_conjugate x(0)
        # warning: t - elapsed time 
        # x0 is the initial configuration of the disease (baseline)
        #xt = self.eigvecs @ np.diag(np.exp(-self.eigvals * self.beta * t)) @ np.conjugate(self.eigvecs.T) @ x0    
           
        step = 1/(self.beta * self.eigvals +1e-5) * (1 - np.exp(-self.beta * self.eigvals * t)) * np.linalg.inv(self.eigvecs + 1e-5) * x0 + self.eigvecs
        xt = x0 + np.sum(step, axis=0) 
        return xt
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for i in range(self.iterations):
            next_step = self.integration_step(diffusion[i], self.timestep)
            diffusion.append(next_step)  
            
        return np.asarray(diffusion)   
    
    def integration_step_by_Julien(self, x_prev, timestep):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        # x(t)/dt = -B * H * x(t)
        #xt = x_prev - timestep * self.beta * self.L @ x_prev
        
        # where x(t) = e^(-B*H*t) * x0
        #xt = x_prev - timestep * self.beta * self.L @ x_prev

        # where x(t) = U * e^(-lambda*B*t) * U^(-1) * x0
        step = self.eigvecs * np.exp(-self.eigvals*self.beta*timestep) * np.linalg.inv(self.eigvecs + 1e-10) @ x_prev
        xt = x_prev - step

        return xt
    
    def iterate_spreading_by_Julien(self):
        diffusion = [self.diffusion_init]  
        
        for i in range(self.iterations):
            next_step = self.integration_step_by_Julien(diffusion[-1], self.timestep)
            diffusion.append(next_step)  
            
        return np.asarray(diffusion) 
 
    def calc_exponent(self):
        ''' Inverse operation to log1p. '''
        self.cm = np.expm1(self.cm)
 
    def downsample_matrix(self, matrix, target_len=int(1e3)):
        ''' Take every n-th sample when the matrix is longer than target length. '''
        current_len = matrix.shape[0]
        if current_len > target_len:
            factor = int(current_len/target_len)
            matrix = matrix[::factor, :] # downsampling
        return matrix
    
    def save_diffusion_matrix(self, save_dir):
        np.savetxt(os.path.join(save_dir, 'diffusion_matrix_over_time.csv'), 
                                self.diffusion_final, delimiter=",")
    
    def save_terminal_concentration(self, save_dir):
        ''' Save last (terminal) concentration. '''
        np.savetxt(os.path.join(save_dir, 'terminal_concentration.csv'),
                   self.diffusion_final[-1, :], delimiter=',')

def run_simulation(dataset_dir, output_dir, subject):    
    ''' Run simulation for single patient. '''
      
    connectivity_matrix_path = os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'dwi', 'connect_matrix_rough.csv'))
    t0_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-baseline', 'pet', '*.csv')))[0]
     # TODO: make it compatible with multiple followup sessions
    t1_concentration_path = glob(os.path.join(os.path.join(dataset_dir, subject, 
                                            'ses-followup', 'pet', '*.csv')))[0]

    subject_output_dir = os.path.join(output_dir, subject)
    
    # load connectome
    connect_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brian regions
    t0_concentration = load_matrix(t0_concentration_path) 
    t1_concentration = load_matrix(t1_concentration_path)

    beta = 1
    step = 1
    min_rmse = -1
    opt_beta = None
    opt_pcc = None
    min_t1_concentration_pred = None
    for _ in tqdm(range(100)):
        simulation = DiffusionSimulation(connect_matrix, beta, t0_concentration)
        t1_concentration_pred = simulation.run()
        simulation.save_diffusion_matrix(subject_output_dir)
        simulation.save_terminal_concentration(subject_output_dir)
        '''
        visualize_diffusion_timeplot(simulation.diffusion_final.T, 
                                    simulation.timestep,
                                    simulation.t_total,
                                    save_dir=subject_output_dir)
        '''
        rmse = calc_rmse(t1_concentration_pred, t1_concentration)
        corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
        if rmse < min_rmse or min_rmse == -1:
            min_rmse = rmse 
            opt_pcc = corr_coef
            min_t1_concentration_pred = t1_concentration_pred
            opt_beta = beta

        beta += step

    logging.info(f'Optimal beta was {opt_beta}')
    logging.info(f'Minimum MSE for subject {subject} is: {min_rmse:.2f}')
    logging.info(f'Corresponding Pearson correlation coefficient for subject {subject} is: {opt_pcc:.2f}')
    
    visualize_terminal_state_comparison(t0_concentration, 
                                        min_t1_concentration_pred,
                                        t1_concentration,
                                        subject,
                                        min_rmse,
                                        opt_pcc,
                                        save_dir=subject_output_dir)
    
def main():
    dataset_dir = '../../data/ADNI/derivatives/'
    output_dir = '../../results' 
    
    patients = ['sub-AD4009', 'sub-AD4215']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(dataset_dir, output_dir, subject)
    
if __name__ == '__main__':
    main()