''' Script for extracting mean amyloid beta concentration 
from brain regions based on atlas. '''

import os
from glob import glob
import logging
import csv

import nibabel
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def load_atlas(path):
    data = nibabel.load(path).get_fdata()
    return data

def get_atlas_labels_info(atlas):
    # count unique values in atlas 
    return np.unique(atlas, return_counts=True)

def load_pet(path):
    pet = nibabel.load(path).get_fdata()
    visualize_PET(pet)
    return pet

def visualize_PET(data):
    plt.imshow(montage(data[:,:,:]), cmap='plasma')
    plt.tight_layout()
    plt.colorbar()
    plt.show()

def extract_regions_means(pet_data, atlas_data):
    means = []
    # do not take atlas region with index=0, which indicates the background
    atlas_labels = [label for label in np.unique(atlas_data) if label!=0]
    for label in np.sort(atlas_labels):
        avg = pet_data[np.where(label == atlas_data)].mean()
        means.append(avg)
    return means

def save_concentrations(concentrations, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(concentrations)

def run(pet_dir, concentrations_dir, subject, atlas_data):
    # get the preprocessed PET data
    pet_files_paths = glob(os.path.join(pet_dir, subject, 
                                        'ses-*', 'pet', '*_pet.nii'))

    for path in pet_files_paths:
        filename = os.path.basename(path).split('.')[0]
        output_dir = os.path.join(concentrations_dir, subject)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, filename+'.csv')
        
        pet_data = load_pet(path)
        pet_all_patients.append(pet_data)
        region_means = extract_regions_means(pet_data, atlas_data)
        save_concentrations(region_means, output_path)
    
def main():
    pet_dir = '../../data/PET'
    concentrations_dir = '../../data/PET_regions_concentrations'
    atlas_path = '../../data/atlas/aal.nii.gz'
    global pet_all_patients
    pet_all_patients = [] # store data from all patients 
    
    atlas_data = load_atlas(atlas_path)
    
    patients = ['sub-AD4009', 'sub-AD4215']
    for subject in patients:
        logging.info(f'Beta-amyloid concentration extraction for subject: {subject}')
        run(pet_dir, concentrations_dir, subject, atlas_data)
        
    print('Are values the same for 2 first subjects?')
    print(np.array_equal(pet_all_patients[:2], pet_all_patients[2:]))

if __name__ == '__main__':
    main()