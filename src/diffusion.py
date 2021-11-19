###################################################################################
########## Spreading model based on Heat-kernel diffusion #########################
########## Warining: It requires: Python 3.4 due to the decorators#################
##########                                                               ##########   
########## Code initially developped by Jimmy Jackson (AIMS-Ghana) and   ##########
########## then further improved by Alessandro Crimi and Matteo Frigo    ##########
###################################################################################   

import os

from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt


############# Parameters
beta = 1.5  #As in the Raj et al. papers
iterations = int(1e3)#1000
rois = 116 #AAL atlas has 116 rois
tstar = 10.0
timestep = tstar / iterations

# Increment done with decorators
def integration_step(x0, t):
    xt = eigvecs.T @ x0
    xt = np.diag(np.exp(-beta * t * eigvals)) @ xt
    return eigvecs @ xt
 
## Set Alzheimer seed regions by hand

#Store initial misfolded proteins
X0 = np.zeros(rois)
# Seed region for Pankinson is the brainstem
# 8,13,97,182 
# Seed regions for Alzheimer (TODO: confirm)
# 31, 32, 35, 36
X0[31] = 1
X0[32] = 1
X0[35] = 1
X0[36] = 1

# load connectivity matrix
path = '../data/output/sub-AD4009/connect_matrix_filtered.csv'
cm_data = np.genfromtxt(path, delimiter=",")

# Compute Laplacian 
adjacency = cm_data
degrees = np.sum(adjacency, axis=1)
dm12 = np.diag(1.0 / np.sqrt(degrees))
laplacian = np.eye(adjacency.shape[0]) - (dm12 @ adjacency) @ dm12
eigvals, eigvecs = np.linalg.eig(laplacian)

# Iterate spreading    
all_steps = [X0]  #List containing all timepoints

for _ in tqdm(range(iterations)):
    next_step = integration_step(all_steps[-1], timestep)
    all_steps.append(next_step)

A = np.asarray(all_steps)

# plot results
plt.figure(figsize=(15,3))
plt.imshow(A.T) #, interpolation='nearest'
plt.xlabel('Iteration' )
plt.ylabel('ROIs' )
plt.colorbar()
plt.tight_layout()
plt.show() 