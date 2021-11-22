''' Spreading model based on Heat-kernel diffusion. '''

import os

from tqdm import tqdm 
import numpy as np

from utils_vis import visualize_diffusion_matrix

class DiffusionSimulation:
    def __init__(self, connect_matrix):
        self.beta = 1.5  #As in the Raj et al. papers
        self.iterations = int(1e5) #1000
        self.rois = 116 #AAL atlas has 116 rois
        self.tstar = 10.0
        self.timestep = self.tstar / self.iterations
        self.cm = connect_matrix
        
    def run(self, downsample=True):
        ''' Run simulation. '''
        self.calc_exponent()
        self.define_seeds()
        self.calc_laplacian()
        self.diffusion_final = self.iterate_spreading()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        
    def define_seeds(self):
        ''' Define Alzheimer seed regions manually. '''
        # Store initial misfolded proteins
        X0 = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        X0[31] = 1
        X0[32] = 1
        X0[35] = 1
        X0[36] = 1
        self.diffusion_init = X0
        
    def calc_laplacian(self):
        adjacency = self.cm
        degrees = np.sum(adjacency, axis=1)
        dm12 = np.diag(1.0 / np.sqrt(degrees))
        laplacian = np.eye(adjacency.shape[0]) - (dm12 @ adjacency) @ dm12
        self.eigvals, self.eigvecs = np.linalg.eig(laplacian)
    
    def integration_step(self, x0, t):
        xt = self.eigvecs.T @ x0
        xt = np.diag(np.exp(-self.beta * t * self.eigvals)) @ xt
        return self.eigvecs @ xt    
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for _ in tqdm(range(self.iterations)):
            next_step = self.integration_step(diffusion[-1], self.timestep)
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
    
    def save_matrix(self, path):
        np.savetxt(path, self.diffusion_final, delimiter=",")
 
def load_connectivity_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def main():
    data_path = '../data/output/sub-AD4009/connect_matrix_filtered.csv'
    output_path = '../data/output/sub-AD4009/diffusion_matrix.csv'
    
    connect_matrix = load_connectivity_matrix(data_path)
    simulation = DiffusionSimulation(connect_matrix)
    simulation.run()
    simulation.save_matrix(output_path)
    
    visualize_diffusion_matrix(simulation.diffusion_final)
    
if __name__ == '__main__':
    main()