''' Spreading model based on Heat-kernel diffusion. 

Based on publication: 
Ashish Raj, Amy Kuceyeski, Michael Weiner,
"A Network Diffusion Model of Disease Progression in Dementia"
'''

import os
import logging
from glob import glob

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian
from scipy.stats.stats import pearsonr as pearson_corr_coef

from utils_vis import visualize_diffusion_timeplot, visualize_terminal_state_comparison
from utils import load_matrix, calc_rmse, calc_msle

logging.basicConfig(level=logging.INFO)

class DiffusionSimulation:
    def __init__(self, connect_matrix, concentrations=None):
        ''' If concentration is not None: use PET data as the initial concentration of the proteins. 
        Otherwise: manually choose initial seeds and concentrations. '''
        
        self.beta = 1.5 # As in the Raj et al. papers
        self.rois = 116 # AAL atlas has 116 rois
        self.t_total = 1 # total length of the simulation in years
        self.timestep = 0.01
        self.iterations = int(self.t_total / self.timestep)
        self.cm = connect_matrix
        if concentrations is not None: 
            logging.info(f'Loading concentration from PET files.')
            self.diffusion_init = concentrations
        else:
            logging.info(f'Loading concentration manually.')
            self.diffusion_init = self.define_seeds()
                    
    def run(self, inverse_log=True, downsample=False):
        ''' Run simulation. '''
        if inverse_log: self.calc_exponent()
        self.calc_laplacian()
        self.diffusion_final = self.iterate_spreading_by_Julien()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        return self.diffusion_final[-1]
        
    def define_seeds(self, init_concentration=1):
        ''' Define Alzheimer seed regions manually. 
        
        Args:
            init_concentration (int): initial concentration of misfolded proteins in the seeds. '''
            
        # Store initial misfolded proteins
        diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        # assign initial concentration of proteins in this region
        diffusion_init[[31, 32, 35, 36]] = init_concentration
        return diffusion_init
        
    def calc_laplacian(self, eps=1e-10): 
        # calculate normalized Laplacian: L = I - D-1/2 @ A @ D-1/2
        # assume: A - adjacency matrix, D - degree matrix, I - identity matrix, L - laplacian matrix
        A = self.cm
        D = np.diag(np.sum(A, axis=1))# total no. of. connections to other vertices
        I = np.identity(A.shape[0]) # identity matrix
        D_inv_sqrt = np.linalg.inv(np.sqrt(D)+eps) # add epsilon to avoid getting 0 determinant
        self.L = I - (D_inv_sqrt @ A) @ D_inv_sqrt
                        
        # eigendecomposition
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)
        
    def integration_step(self, x0, t):
        # persistent mode of propagation
        # x(t) = U exp(-lambda * beta * t) U_conjugate x(0)
        # warning: t - elapsed time 
        # TODO: x0 should be the initial concentration or previous concentration?   
        xt = self.eigvecs @ np.diag(np.exp(-self.eigvals * self.beta * t)) @ np.conjugate(self.eigvecs.T) @ x0        
        return xt
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for i in tqdm(range(self.iterations)):
            next_step = self.integration_step(diffusion[0], i*self.timestep)
            diffusion.append(next_step)  
            
        return np.asarray(diffusion)   
    
    def integration_step_by_Julien(self, x_prev, timestep):
        # methods proposed by Julien Lefevre during Marseille Brainhack 
        xt = x_prev - timestep * self.beta * self.L @ x_prev
        return xt
    
    def iterate_spreading_by_Julien(self):
        diffusion = [self.diffusion_init]  
        
        for i in tqdm(range(self.iterations)):
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

def run_simulation(connectomes_dir, concentrations_dir, output_dir, subject):    
    ''' Run simulation for single patient. '''
      
    connectivity_matrix_path = os.path.join(os.path.join(connectomes_dir, subject), 
                                            'connect_matrix_rough.csv')
    concentrations_paths = glob(os.path.join(concentrations_dir, subject, '*.csv'))
    t0_concentration_path = [path for path in concentrations_paths if 'baseline' in path][0]
    t1_concentration_path = [path for path in concentrations_paths if 'followup' in path][0]
    subject_output_dir = os.path.join(output_dir, subject)
    
    # load connectome
    connect_matrix = load_matrix(connectivity_matrix_path)
    # load proteins concentration in brian regions
    t0_concentration = load_matrix(t0_concentration_path) 
    t1_concentration = load_matrix(t1_concentration_path)
            
    simulation = DiffusionSimulation(connect_matrix, t0_concentration)
    t1_concentration_pred = simulation.run()
    simulation.save_diffusion_matrix(subject_output_dir)
    simulation.save_terminal_concentration(subject_output_dir)
    visualize_diffusion_timeplot(simulation.diffusion_final.T, 
                                 simulation.timestep,
                                 simulation.t_total,
                                 save_dir=subject_output_dir)
    rmse = calc_rmse(t1_concentration_pred, t1_concentration)
    corr_coef = pearson_corr_coef(t1_concentration_pred, t1_concentration)[0]
    logging.info(f'MSE for subject {subject} is: {rmse:.2f}')
    logging.info(f'Pearson correlation coefficient for subject {subject} is: {corr_coef:.2f}')
    
    visualize_terminal_state_comparison(t0_concentration, 
                                        t1_concentration_pred,
                                        t1_concentration,
                                        subject,
                                        rmse,
                                        corr_coef,
                                        save_dir=subject_output_dir)
    
def main():
    connectomes_dir = '../../data/connectomes'
    concentrations_dir = '../../data/PET_regions_concentrations'
    output_dir = '../../results' 
    
    patients = ['sub-AD4215', 'sub-AD4009']
    for subject in patients:
        logging.info(f'Simulation for subject: {subject}')
        run_simulation(connectomes_dir, concentrations_dir, output_dir, subject)
    
if __name__ == '__main__':
    main()