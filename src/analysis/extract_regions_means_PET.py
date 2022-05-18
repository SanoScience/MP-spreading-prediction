''' Script for extracting mean amyloid beta concentration 
from brain regions based on atlas. '''

from collections import defaultdict
import os
from glob import glob
import logging
import csv
from statistics import mean
from tqdm import tqdm

import nibabel
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
import multiprocessing
import yaml
from datetime import datetime


def load_atlas(path):
    data = nibabel.load(path).get_fdata()
    return data

def get_atlas_labels_info(atlas):
    # count unique values in atlas 
    return np.unique(atlas, return_counts=True)

def emptiness_test(path, pet, results_file_path="../../results/analysis/empty_pets.txt"):
    ''' Check if loaded PET data contains only zeros. '''
    if np.all(pet == 0): 
        logging.info(f'Zero concentrations for {path}')
        with open(results_file_path, "a+") as f:
            f.write(f'{path}\n')
    return np.all(pet==0)

def load_pet(path, visualize=False):
    pet = nibabel.load(path).get_fdata()
    if visualize: visualize_PET(pet)
    return pet

def visualize_PET(data):
    plt.imshow(montage(data), cmap='plasma')
    plt.tight_layout()
    plt.colorbar()
    plt.show()

def extract_regions_means(pet, pet_data, atlas_data):
    means = []
    # do not take atlas region with index=0, which indicates the background
    # however, in AAL3v1_1mm there is no label = 0 
    # instead, there are 166 unique labels (35, 36, 81, 82 are missing)
    atlas_labels = [label for label in np.unique(atlas_data) if label!=0]
    assert len(atlas_labels) == 166
    
    for label in atlas_labels:
        try:
            avg = pet_data[np.where(label == atlas_data)].mean()
        except Exception as e:
            logging.erro(f"Invalid index for image {pet}")
            avg = 0
        means.append(avg)
    
    # normalize within the PET (divide by maximum value)
    max_val = max(means)
    for v in means:
        v /= max_val
        if v<0 or v>1: logging.error(f"Image {pet} with invalid value {v}")
        
    return means

def save_concentrations(concentrations, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(concentrations)
    logging.info(f'Extracted concentrations saved in {path}')

def run(pet, atlas_data, q):    
    pet_data = load_pet(pet)
    if emptiness_test(pet, pet_data): return
    region_means = extract_regions_means(pet, pet_data, atlas_data)
    q.put_nowait((pet, region_means))
    return
    

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

if __name__ == '__main__':
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.chdir(os.getcwd() + '/../../..')
    general_dir = os.getcwd() + os.sep
    logging.info(general_dir)
    dataset_dir = general_dir + config['paths']['dataset_dir'] + 'sub-*'  + os.sep + 'ses-*' + os.sep + 'pet' + os.sep + 'sub*pet.nii.gz'
    logging.info(dataset_dir)

    atlas_path = general_dir + config['paths']['atlas_path']
    atlas_data = load_atlas(atlas_path)
    
    num_cores = ''
    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
    except Exception as e:
        num_cores = multiprocessing.cpu_count()

    logging.info(f"{num_cores} cores available")

    pets = glob(dataset_dir)
    q = multiprocessing.Queue()
    procs = []
    counter = 0  
    done = 0
    concentrations = {}
    cn_sum = []
    counter = 0
    
    for i in tqdm(range(len(pets))):
        logging.info(f'Beta-amyloid concentration extraction in image: {pets[i]}')
        p = multiprocessing.Process(target=run, args=(pets[i], atlas_data, q))
        p.start()
        procs.append(p)
        counter +=1
        if counter%4 == 0 and counter > 0:
            pet, regions = q.get()      
            concentrations[pet] = np.array(regions)
            if 'sub-CN' in pet:
                cn_sum.append(regions)
            counter -= 1
            done += 1
    while done < len(procs):
        pet, regions = q.get()      
        concentrations[pet] = np.array(regions)
        if 'sub-CN' in pet:
            cn_sum.append(regions)
        done += 1
        
    logging.info("Z-score normalization")
    cn_mean = np.mean(cn_sum, axis=0)
    cn_std = np.std(cn_sum, axis=0)
    
    for pet in concentrations.keys():
        # z-score (region(i) - healthy_mean(i))/healthy_std(i)
        concentrations[pet] = (concentrations[pet] - cn_mean)/cn_std
        
        # sigmoid normalization 1/(e^(-region(i)) + e^(region(i)))
        concentrations[pet] = 1/(np.exp(-1*concentrations[pet]) + np.exp(concentrations[pet]))  
        
        output_path = pet.replace('.nii.gz', '.csv')          
        save_concentrations(concentrations[pet], output_path)