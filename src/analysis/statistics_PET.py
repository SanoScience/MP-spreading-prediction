''' Script for calculating beta-amyloid concentration statistics. '''

import os
from glob import glob
import logging
import csv

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def boxplot(data, title=None, xlabel=None, labels=None, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    bp = plt.boxplot(data, showfliers=False, showmeans=True, notch=False, labels=labels)
    plt.ylabel('amyloid-beta concentration')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend([bp['medians'][0], bp['means'][0]], ['medians', 'means'])
    plt.tight_layout()
    plt.show()

def run(concentrations_dir, subjects):
    # get concentration data (index=0 (background) not included)
    files_paths = [glob(os.path.join(concentrations_dir, subject, 
                                        '*.csv')) for subject in subjects]
    data = np.array([np.genfromtxt(path[0], delimiter=',') for path in files_paths])

    region_means = np.mean(data, axis=0)
    patient_mean = np.mean(data, axis=1)
    dataset_mean = np.mean(data)
    
    boxplot(data, 'region concentration averaged by patients', 'region index', figsize=(25,5))
    boxplot(np.swapaxes(data, 0, 1), 'concentration averaged by regions', 
            'patient index', labels=subjects, figsize=(10,10))
    
def main():
    concentrations_dir = '../../data/PET_regions_concentrations'   
    patients = ['sub-AD4009', 'sub-AD4215', 'sub-AD4500', 'sub-AD4892', 'sub-AD6264']
    run(concentrations_dir, patients)

if __name__ == '__main__':
    main()