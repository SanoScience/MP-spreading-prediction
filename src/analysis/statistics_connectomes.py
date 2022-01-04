''' Script for calculating connection statistics based on connectivity matrix. '''

import os
from glob import glob
import logging
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

logging.basicConfig(level=logging.INFO)
    
def heatmap(array, binary=False):
    plt.figure(figsize=(15, 12))
    if binary:
        cmap = matplotlib.colors.ListedColormap(['w', 'b'])
        plt.imshow(array, cmap=cmap)
        plt.colorbar(ticks=[0, 1])
        plt.title('Non-negative values above the mean')
    else: 
        plt.imshow(array)
        plt.title('Mean connection strength across dataset (log values: True)')
        plt.colorbar()
    plt.tight_layout()
    plt.show()
    
def count_values_above_mean(array, mean):
    # take into account only non-negative values
    condition_matrix = np.logical_and(array > mean, array >= 0)
    counts = condition_matrix.sum()
    return condition_matrix, counts

def run(connectomes_dir, subjects):
    # get connetomes data (not inverted)
    files_paths = [glob(os.path.join(connectomes_dir, subject, 
                                        '*rough.csv')) for subject in subjects]
    # load data
    data = np.array([np.genfromtxt(path[0], delimiter=',') for path in files_paths])
    # remove cerebellum and background
    data = [array[1:91, 1:91] for array in data]
    
    print('Mean value of connection strength across whole dataset')
    mean_across_subj = np.mean(data, axis=0)
    heatmap(mean_across_subj)
    
    print('No. of non-negative values above mean:')
    for array, subj in zip(data, subjects):
        condition_matrix, counts = count_values_above_mean(array, mean_across_subj)
        ratio = counts/array.shape[0]**2
        print(f'Subject: {subj}, counts: {counts}, ratio: {ratio:.2f}')
        heatmap(condition_matrix, binary=True)
        
def main():
    connectomes_dir = '../../data/connectomes'   
    patients = ['sub-AD4009', 'sub-AD4215', 'sub-AD4500', 'sub-AD4892', 'sub-AD6264']
    run(connectomes_dir, patients)

if __name__ == '__main__':
    main()