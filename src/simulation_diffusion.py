''' Spreading model based on Heat-kernel diffusion. '''

import os

from tqdm import tqdm 
import numpy as np
from scipy.sparse.csgraph import laplacian as scipy_laplacian

from utils_vis import visualize_diffusion_timeplot

class DiffusionSimulation:
    def __init__(self, connect_matrix):
        self.beta = 1.5 #As in the Raj et al. papers
        self.iterations = int(1e3) #1000
        self.rois = 116 #AAL atlas has 116 rois
        self.tstar = 10.0
        self.timestep = self.tstar / self.iterations
        self.cm = connect_matrix
        
    def run(self, inverse_log=True, downsample=True):
        ''' Run simulation. '''
        if inverse_log: self.calc_exponent()
        self.define_seeds()
        self.calc_laplacian()
        self.diffusion_final = self.iterate_spreading()
        if downsample: 
            self.diffusion_final = self.downsample_matrix(self.diffusion_final)
        
    def define_seeds(self):
        ''' Define Alzheimer seed regions manually. '''
        # Store initial misfolded proteins
        self.diffusion_init = np.zeros(self.rois)
        # Seed regions for Alzheimer (according to AAL atlas): 31, 32, 35, 36 (TODO: confirm)
        self.diffusion_init[[31, 32, 35, 36]] = 1
        
    def calc_laplacian(self):
        # normed laplacian 
        adjacency = self.cm
        degrees = np.sum(adjacency, axis=1) # total no. of. connections to other vertices
        dm12 = np.diag(1.0 / np.sqrt(degrees))
        laplacian1 = np.eye(adjacency.shape[0]) - (dm12 @ adjacency) @ dm12 
        
        adjacency_matrix = self.cm
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1)) # total no. of. connections to other vertices
        laplacian2= degree_matrix - adjacency_matrix
        
        laplacian3 = scipy_laplacian(adjacency, normed=True)
        
        print(np.allclose(laplacian1, laplacian2))
        print(np.allclose(laplacian1, laplacian3))
        print(np.allclose(laplacian2, laplacian3))

        self.eigvals, self.eigvecs = np.linalg.eig(laplacian3)
    
    def integration_step(self, x0, t):
        xt = self.eigvecs.T @ x0
        xt = np.diag(np.exp(-self.beta * t * self.eigvals)) @ xt
        return self.eigvecs @ xt  
    
    def integration_step_vol2(self, x0, t):
        xt = self.eigvecs.T @ x0
        
        # print(np.allclose(np.linalg.inv(self.eigvecs), self.eigvecs.T))
        
        aux_var = (np.ones(len(self.eigvals[1:])) - np.exp(-self.beta * t * self.eigvals[1:])) / (self.beta * self.eigvals[1:])
        d = np.insert(aux_var, 0, t)
        # print(xt.shape, aux_var.shape, len(d), self.eigvecs.shape)
        d  = np.diag(d) @ xt
        return self.eigvecs @ d 
    
    def iterate_spreading(self):  
        diffusion = [self.diffusion_init]  #List containing all timepoints

        for _ in tqdm(range(self.iterations)):
            next_step = self.integration_step_vol2(diffusion[-1], self.timestep)
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
    
    visualize_diffusion_timeplot(simulation.diffusion_final)
    
if __name__ == '__main__':
    main()