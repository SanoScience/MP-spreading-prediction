''' Script for extracting mean amyloid beta concentration 
from brain regions based on atlas. '''

import os
from glob import glob
import logging
import csv
from tqdm import tqdm

import nibabel
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
import multiprocessing

logging.basicConfig(level=logging.INFO)

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

def extract_regions_means(pet_data, atlas_data):
    means = []
    # do not take atlas region with index=0, which indicates the background
    # however, in AAL3v1_1mm there is no label = 0 
    # instead, there are 166 unique labels (35, 36, 81, 82 are missing)
    atlas_labels = [label for label in np.unique(atlas_data) if label!=0]
    assert len(atlas_labels) == 166
    
    for label in atlas_labels:
        avg = pet_data[np.where(label == atlas_data)].mean()
        means.append(avg)
    return means

def save_concentrations(concentrations, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(concentrations)

def run(dataset_dir, subject, atlas_data):
    # get the preprocessed PET data
    pet_files_paths = glob(os.path.join(dataset_dir, subject, 
                                        'ses-*', 'pet', '*_pet.nii.gz'))
    
    for path in pet_files_paths:
        # logging.info(f'Found pet file {path}')   
        output_path = path.replace('.nii.gz', '.csv')
               
        pet_data = load_pet(path)
        if emptiness_test(path, pet_data): continue
        region_means = extract_regions_means(pet_data, atlas_data)
        save_concentrations(region_means, output_path)
        logging.info(f'Extracted concentrations saved in {output_path}')
    
def main():
    dataset_dir = '../../data/ADNI/derivatives/'
    atlas_path = '../../data/atlas/AAL3v1.nii.gz'
    
    atlas_data = load_atlas(atlas_path)
    
    ## try to run it in parallel
    # logger = logging.getLogger()
    # logger.setLevel(logging.ERROR)
    
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     patients = os.listdir(dataset_dir)
    #     for subj in patients:
    #         logging.info(f'Beta-amyloid concentration extraction for subject: {subj}')
    #         executor.submit(run, dataset_dir, subj, atlas_data)
    num_cores = ''
    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
    except Exception as e:
        num_cores = multiprocessing.cpu_count()

    logging.info(f"{num_cores} cores available")

    patients = os.listdir(dataset_dir)
    procs = []
    for subj in tqdm(patients):
        logging.info(f'Beta-amyloid concentration extraction for subject: {subj}')
        p = multiprocessing.Process(target=run, args=(dataset_dir, subj, atlas_data))
        p.start()
        procs.append(p)
        
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                # wait for 10 seconds to wait process termination
                p.join(timeout=10)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
        
        # wait the last chunk            
        for p in procs:
            p.join() 
    
if __name__ == '__main__':
    main()