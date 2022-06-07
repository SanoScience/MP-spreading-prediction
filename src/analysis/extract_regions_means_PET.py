''' Script for extracting mean amyloid beta concentration 
from brain avg_region based on atlas. '''

from collections import defaultdict
import os
from glob import glob
import logging
import csv
from statistics import mean
from tqdm import tqdm
from nibabel import Nifti1Image, save, load
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
import multiprocessing
import yaml
from datetime import datetime

def load_atlas(path):
    atlas = load(path)
    return atlas

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
    pet = load(path).get_fdata()
    if visualize: visualize_PET(pet)
    return pet

def visualize_PET(data):
    plt.imshow(montage(data), cmap='plasma')
    plt.tight_layout()
    plt.colorbar()
    plt.show()

def extract_avg_region_means(pet, pet_data, atlas):
    means = []
    # do not take atlas region with index=0, which indicates the background
    # however, in AAL3v1_1mm there is no label = 0 
    # instead, there are 166 unique labels (35, 36, 81, 82 are missing)
    atlas_labels = [label for label in np.unique(atlas) if label!=0]
    assert len(atlas_labels) == 166
    for label in atlas_labels:
        try:
            indices = np.where(atlas == label)
            avg = pet_data[indices].mean()
            pet_data[indices] = avg
        except Exception as e:
            logging.error(f"Invalid index for image {pet}")
            avg = 0
        means.append(avg)
    
    # put what is not in the atlas to 0 (i.e. skull)
    try:
        pet_data[np.where(atlas==0)] = 0
    except Exception as e:
        logging.error(f"Error during background remotion. Traceback: {e}")
        
    return means, pet_data

def save_concentrations(concentrations, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(concentrations)

def run(pet, atlas, q):    
    pet_data = load_pet(pet)
    if emptiness_test(pet, pet_data): return
    region_means, pet_data = extract_avg_region_means(pet, pet_data, atlas)
    q.put_nowait((pet, region_means, pet_data))
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
    average_images_dir = general_dir + config['paths']['dataset_dir'] + 'datasets/average_images/'
    if not os.path.exists(average_images_dir): os.makedirs(average_images_dir)

    atlas_path = general_dir + config['paths']['atlas_pets']
    atlas = load_atlas(atlas_path)
    
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
    
    ad_region = []
    lmci_region = []
    mci_region = []
    emci_region = []
    cn_region = []
    
    ad_voxels = []
    lmci_voxels = []
    mci_voxels = []
    emci_voxels = []
    cn_voxels = []
    
    counter = 0
    
    for i in tqdm(range(len(pets))):
        p = multiprocessing.Process(target=run, args=(pets[i], atlas.get_fdata(), q))
        p.start()
        procs.append(p)
        counter +=1
        if counter%num_cores == 0 and counter > 0:
            pet, avg_region, voxels_avg = q.get()    
            save(Nifti1Image(voxels_avg, affine=atlas.affine), pet.replace('.nii.gz', '_avg.nii.gz'))  
            concentrations[pet] = np.array(avg_region)
            if 'sub-AD' in pet:
                ad_region.append(avg_region)
                ad_voxels.append(voxels_avg)
            elif 'sub-LMCI' in pet:
                lmci_region.append(avg_region)
                lmci_voxels.append(voxels_avg)
            elif 'sub-MCI' in pet:
                mci_region.append(avg_region)
                mci_voxels.append(voxels_avg)
            elif 'sub-EMCI' in pet:
                emci_region.append(avg_region)
                emci_voxels.append(voxels_avg)
            elif 'sub-CN' in pet:
                cn_region.append(avg_region)
                cn_voxels.append(voxels_avg)
            counter -= 1
            done += 1
    while done < len(procs):
        pet, avg_region, voxels_avg = q.get()      
        concentrations[pet] = np.array(avg_region)
        if 'sub-AD' in pet:
            ad_region.append(avg_region)
            ad_voxels.append(voxels_avg)
        elif 'sub-LMCI' in pet:
            lmci_region.append(avg_region)
            lmci_voxels.append(voxels_avg)
        elif 'sub-MCI' in pet:
            mci_region.append(avg_region)
            mci_voxels.append(voxels_avg)
        elif 'sub-EMCI' in pet:
            emci_region.append(avg_region)
            emci_voxels.append(voxels_avg)
        elif 'sub-CN' in pet:
            cn_region.append(avg_region)
            cn_voxels.append(voxels_avg)
        done += 1
    
    cn_mean = np.mean(cn_region, axis=0)
    cn_std = np.std(cn_region, axis=0)
    
    max_voxel_val = np.max([cn_voxels, emci_voxels, mci_voxels, lmci_voxels, ad_voxels])
    
    save(Nifti1Image(np.mean(ad_voxels / max_voxel_val, axis = 0, dtype=np.float64), affine=atlas.affine), average_images_dir + 'AD.nii.gz')
    print("Saved AD.nii.gz brain image")
    
    save(Nifti1Image(np.mean(lmci_voxels / max_voxel_val, axis = 0, dtype=np.float64), affine=atlas.affine), average_images_dir + 'LMCI.nii.gz')
    print("Saved LCMI.nii.gz brain image")
    
    save(Nifti1Image(np.mean(mci_voxels / max_voxel_val, axis = 0, dtype=np.float64), affine=atlas.affine), average_images_dir + 'MCI.nii.gz')
    print("Saved MCI.nii.gz brain image")
    
    save(Nifti1Image(np.mean(emci_voxels / max_voxel_val, axis = 0, dtype=np.float64), affine=atlas.affine), average_images_dir + 'EMCI.nii.gz')
    print("Saved EMCI.nii.gz brain image")
    
    save(Nifti1Image(np.mean(cn_voxels / max_voxel_val, axis = 0, dtype=np.float64), affine=atlas.affine), average_images_dir + 'CN.nii.gz')
    print("Saved CN.nii.gz brain image")
    

    logging.info(f'Average values across avg_region in CN subjects: {cn_mean}')
    logging.info(f'Std across avg_region in CN subjects: {cn_std}')
    np.savetxt('cn_mean.csv', cn_mean, delimiter=', ')
    np.savetxt('cn_std.csv', cn_std, delimiter=', ')
    
    logging.info("Z-score normalization")
    for pet in concentrations.keys():
        # z-score (region(i) - healthy_mean(i))/healthy_std(i)
        concentrations[pet] = (concentrations[pet] - cn_mean)/cn_std
        
        # sigmoid normalization 1/(e^(-region(i)) + e^(region(i)))
        concentrations[pet] = 1/(np.exp(-1*concentrations[pet]) + np.exp(concentrations[pet]))  
        
        output_path = pet.replace('.nii.gz', '.csv')          
        save_concentrations(concentrations[pet], output_path)
    
    logging.info('*** Done ***')
