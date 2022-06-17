''' Script for extracting mean amyloid beta concentration 
from brain avg_region based on atlas. '''

from collections import defaultdict
import os
from glob import glob
import logging
import csv
import time
from tqdm import tqdm
from nibabel import Nifti1Image, save, load
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
from threading import Thread, Lock
from multiprocessing import cpu_count
import yaml
from datetime import datetime

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force=True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

def load_atlas(path):
    atlas = load(path)
    data = atlas.get_fdata()
    np.nan_to_num(data, copy=False, nan=0, posinf=0, neginf=0)
    return data, atlas.affine

def get_atlas_labels_info(atlas):
    # count unique values in atlas 
    return np.unique(atlas, return_counts=True)

def visualize_PET(data):
    plt.imshow(montage(data), cmap='plasma')
    plt.tight_layout()
    plt.colorbar()
    plt.show()

def save_concentrations(concentrations, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(concentrations)

class pet_loader(Thread):
    def __init__(self, pet):
        Thread.__init__(self)
        self.pet = pet 

    def load_pet(self):
        pet_data = load(self.pet).get_fdata()
        if np.sum(pet_data) == 0:
            print(f"File {self.pet} is empty!")
            logging.error(f"File {self.pet} is empty!")
        #pet_data /= np.max(pet_data)
        return pet_data

    # NOTE: this function override the homonyms 'run' in the Thread class, executed when the method 'start' is invoked
    def run(self):  
        pet_data = self.load_pet()
        
        id = self.pet.split('/')[-1].split('.')[0].split('_')[0]
        if 'ses-baseline' in self.pet:
            session = 'baseline'
        else:
            session = 'followup'
            
        lock.acquire()        
        subj_pet[id].append([session, pet_data])
        lock.release()

class pet_average(Thread):
    def __init__(self, file_id, pet_id, pet_data, max_val):
        Thread.__init__(self)
        self.file_id = file_id
        self.pet_id = pet_id
        self.pet_data = pet_data
        self.max_val = max_val
        
    def extract_avg_region_means(self):
        means = []
        # do not take atlas region with index=0, which indicates the background
        # there are 166 unique labels (35, 36, 81, 82 are missing)
        atlas_labels = [label for label in np.unique(atlas_data) if label>0]
        assert len(atlas_labels) == 166
        for label in atlas_labels:
            try:
                # NOTE: storing indices doesn't work! (see below)
                # indices = np.where(atlas == label)
                avg = self.pet_data[np.where(atlas_data == label)].mean()
                self.pet_data[np.where(atlas_data == label)] = avg
            except Exception as e:
                logging.error(f"Invalid index for image {self.pet_id}")
                print(f"Invalid index for image {self.pet_id}")
                avg = 0
            means.append(avg)
        
        # put what is not in the atlas to 0 (i.e. skull)
        try:
            self.pet_data[np.where(atlas_data==0)] = 0
            np.nan_to_num(self.pet_data, copy=False, nan=0, posinf=0, neginf=0)
        except Exception as e:
            logging.error(f"Error during background remotion. Traceback: {e}")
            print(f"Error during background remotion. Traceback: {e}")
        
        # return mean concentration for each region and the new voxels with value corresponding to correspondent regional concentration
        return np.array(np.mean(means, axis=0))

    def run(self):
        self.pet_data /= self.max_val
        means = self.extract_avg_region_means()
        lock.acquire()
        voxels[self.file_id] = means
        if 'sub-AD' in self.pet_id:
            ad_voxels.append(means)
        elif 'sub-LMCI' in self.pet_id:
            lmci_voxels.append(means)
        elif 'sub-MCI' in self.pet_id:
            mci_voxels.append(means)
        elif 'sub-EMCI' in self.pet_id:
            emci_voxels.append(means)
        if 'sub-CN' in self.pet_id:      
            cn_voxels.append(means)
            if 'ses-baseline' in self.pet_id:
                cn_baseline_region.append(means)
                cn_baseline_voxels.append(means)
        lock.release()

if __name__ == '__main__':
    ## INPUT
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
    atlas_data, atlas_affine = load_atlas(atlas_path)

    num_cores = ''
    try:
        num_cores = int(input('Cores to use [hit \'Enter\' for all available]: '))
    except Exception as e:
        num_cores = cpu_count()
        logging.info(f"{num_cores} cores available")

    pets = glob(dataset_dir)
    
    # NOTE: the following variables will be visible inside the Thread without needing for importing! They will always point to the same structure    
    subj_pet = defaultdict(list)
    concentrations = {}
    voxels = {}
    
    ad_voxels = []
    lmci_voxels = []
    mci_voxels = []
    emci_voxels = []
    cn_voxels = []
    cn_baseline_voxels = []
    cn_baseline_region = []
    lock = Lock()     
    works = []

    ### PET READING
    print('Reading PETs...')
    for i in tqdm(range(len(pets))):
        works.append(pet_loader(pets[i]))
        works[-1].start()

        while len(works) >= num_cores:
            for w in works:
                w.join(timeout=0.1)
                if not w.is_alive():
                    works.remove(w)
    
    for w in works:
        w.join()
        works.remove(w)
        
    
    ### NORMALIZE VALUES USING THE MAXIMUM VALUE BETWEEN BASELINE AND FOLLOWUP
    print('Normalizing PETs...')
    for subj_id in tqdm(subj_pet):
        max_val = 0
        for data in subj_pet[subj_id]:
            tmp = np.max(data[1])
            if max_val < tmp:
                max_val = tmp
                
        for data in subj_pet[subj_id]:
            file_id = ''
            for p in pets:
                if data[0] in p and subj_id in p:
                    file_id = p
                    break
            works.append(pet_average(file_id, subj_id, data[1], max_val))
            works[-1].start()
        while len(works) >= num_cores:
            for w in works:
                if not w.is_alive():
                    works.remove(w)
    
    for w in works:
        w.join()
        works.remove(w)   
    
    time.sleep(3)
    
    ### AVERAGE PET VOXELS
    print('Averaging PETs voxels...')
    try:
        ad_voxels = np.array(ad_voxels)
        ad_voxels = np.mean(ad_voxels, axis=0, dtype=np.float64)
        ad_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Exception during averaging of AD PETs voxels. Traceback: {e}")
        print(f"Exception during averaging of AD PETs voxels. Traceback: {e}")
    try:
        lmci_voxels = np.array(lmci_voxels)
        lmci_voxels = np.mean(lmci_voxels, axis=0, dtype=np.float64)
        lmci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Exception during averaging of LMCI PETs voxels. Traceback: {e}")
        print(f"Exception during averaging of LMCI PETs voxels. Traceback: {e}")
    try:
        mci_voxels = np.array(mci_voxels)
        mci_voxels = np.mean(mci_voxels, axis=0, dtype=np.float64)
        mci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Exception during averaging of MCI PETs voxels. Traceback: {e}")
        print(f"Exception during averaging of MCI PETs voxels. Traceback: {e}")
    try:
        emci_voxels = np.array(emci_voxels)
        emci_voxels = np.mean(emci_voxels, axis=0, dtype=np.float64)
        emci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Exception during averaging of EMCI PETs voxels. Traceback: {e}")
        print(f"Exception during averaging of EMCI PETs voxels. Traceback: {e}")
    try:
        cn_voxels = np.array(cn_voxels)
        cn_voxels = np.mean(cn_voxels, axis=0, dtype=np.float64)
        cn_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Exception during averaging of CN PETs voxels. Traceback: {e}")
        print(f"Exception during averaging of CN PETs voxels. Traceback: {e}")
        
    # SAVING NIFTI
    try:
        save(Nifti1Image(ad_voxels, affine=atlas_affine), average_images_dir + 'AD.nii.gz')
        print("Saved AD.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving AD average concentrations. Traceback: {e}")
        print(f"Error in saving AD average concentrations. Traceback: {e}")
    try:
        save(Nifti1Image(lmci_voxels, affine=atlas_affine), average_images_dir + 'LMCI.nii.gz')
        print("Saved LMCI.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving LMCI average concentrations. Traceback: {e}")
        print(f"Error in saving LMCI average concentrations. Traceback: {e}")
    try:                
        save(Nifti1Image(mci_voxels, affine=atlas_affine), average_images_dir + 'MCI.nii.gz')
        print("Saved MCI.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving MCI average concentrations. Traceback: {e}")
        print(f"Error in saving MCI average concentrations. Traceback: {e}")
    try:                
        save(Nifti1Image(emci_voxels, affine=atlas_affine), average_images_dir + 'EMCI.nii.gz')
        print("Saved EMCI.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving EMCI average concentrations. Traceback: {e}")
        print(f"Error in saving EMCI average concentrations. Traceback: {e}")
    try:                
        save(Nifti1Image(cn_voxels, affine=atlas_affine), average_images_dir + 'CN.nii.gz')
        print("Saved CN.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving CN average concentrations. Traceback: {e}")
        print(f"Error in saving CN average concentrations. Traceback: {e}")
        
    ### CN BASELINE CONCENTRATIONS 
    print("Computing CN baseline average concentrations...")
    try:
        # REGIONS
        cn_baseline_region = np.array(cn_baseline_region)
        
        cn_baseline_mean = np.mean(cn_baseline_region, axis=0, dtype=np.float64)
        
        cn_baseline_std = np.std(cn_baseline_region, axis=0, dtype=np.float64)
        cn_baseline_std[cn_baseline_std == 0] = 1
        
        np.savetxt('cn_baseline_mean.csv', cn_baseline_mean, delimiter=', ')
        np.savetxt('cn_baseline_std.csv', cn_baseline_std, delimiter=', ')

        # VOXELS
        cn_baseline_voxels = np.array(cn_baseline_voxels)
        
        cn_voxels_mean = np.mean(cn_baseline_voxels, axis = 0, dtype=np.float64)
        cn_voxels_mean[np.where(atlas_data==0)] = 0
        
        cn_voxels_std = np.std(cn_baseline_voxels, axis = 0, dtype=np.float64)
        cn_voxels_std[cn_voxels_std == 0] = 1
        cn_voxels_std[np.where(atlas_data==0)] = 1 
    except Exception as e:
        logging.error(f"Error during computation of CN baseline average concentrations. Traceback: {e}")
        print(f"Error during computation of CN baseline average concentrations. Traceback: {e}")
        cn_baseline_mean = 0
        cn_baseline_std = 1
        cn_voxels_mean = 0
        cn_voxels_std = 1
            
    ### NORMALIZATION OF SUBJECTS CONCENTRATIONS
    logging.info("Z-score normalization of subjects regional concentrations")
    print("Computing z-score normalization of subjects regional concentrations")
    for s in tqdm(concentrations.keys()):
        try:
            # TODO: Modify this and test
            # z-score (region(i) - healthy_mean(i))/healthy_std(i)
            #concentrations[s] = (concentrations[s] - cn_baseline_mean) / cn_baseline_std
            
            # standard logistic function e^(region[i]) / ( e^(region(i)) + 1 )
            #concentrations[s] = 1 / (1 + np.exp(- concentrations[s]))  
            
            output_path = s.replace('.nii.gz', '.csv')          
            save_concentrations(concentrations[s], output_path)
        except Exception as e:
            logging.error(f"Error in normalization of subject {s} regional concentrations. Traceback: {e}")
            print(f"Error in normalization of subject {s} regional concentrations. Traceback: {e}")
    
    
    ### NORMALIZATION OF CATEGORY VOXELS CONCENTRATIONS
    logging.info("Z-score normalization of category concentrations")
    print("Z-score normalization of category concentrations")
    try:
        ad_voxels = (ad_voxels - cn_voxels_mean) / cn_voxels_std
        ad_voxels = 1 / (1 + np.exp(- ad_voxels))
        ad_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Error in Z-score normalization of AD voxels concentrations. Traceback: {e}")
        print(f"Error in Z-score normalization of AD voxels concentrations. Traceback: {e}")
    try:
        lmci_voxels = (lmci_voxels - cn_voxels_mean) / cn_voxels_std
        lmci_voxels = 1 / (1 + np.exp(- lmci_voxels)) 
        lmci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Error in Z-score normalization of LMCI voxels concentrations. Traceback: {e}")
        print(f"Error in Z-score normalization of LMCI voxels concentrations. Traceback: {e}")
    try:    
        mci_voxels = (mci_voxels - cn_voxels_mean) / cn_voxels_std
        mci_voxels = 1 / (1 + np.exp(- mci_voxels))
        mci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Error in Z-score normalization of MCI voxels concentrations. Traceback: {e}")
        print(f"Error in Z-score normalization of MCI voxels concentrations. Traceback: {e}")
    try:    
        emci_voxels = (emci_voxels - cn_voxels_mean) / cn_voxels_std
        emci_voxels = 1 / (1 + np.exp(- emci_voxels))
        emci_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Error in Z-score normalization of EMCI voxels concentrations. Traceback: {e}")
        print(f"Error in Z-score normalization of EMCI voxels concentrations. Traceback: {e}")
    try:    
        cn_voxels = (cn_voxels - cn_voxels_mean) / cn_voxels_std
        cn_voxels = 1 / (1 + np.exp(- cn_voxels))
        cn_voxels[np.where(atlas_data==0)] = 0
    except Exception as e:
        logging.error(f"Error in Z-score normalization of CN voxels concentrations. Traceback: {e}")
        print(f"Error in Z-score normalization of CN voxels concentrations. Traceback: {e}")


    try:        
        save(Nifti1Image(ad_voxels, affine=atlas_affine), average_images_dir + 'AD_norm.nii.gz')
        print("Saved AD_norm.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving AD normalized concentrations. Traceback: {e}")
        print(f"Error in saving AD normalized concentrations. Traceback: {e}")

    try:        
        save(Nifti1Image(lmci_voxels, affine=atlas_affine), average_images_dir + 'LMCI_norm.nii.gz')
        print("Saved LMCI_norm.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving LMCI normalized concentrations. Traceback: {e}")
        print(f"Error in saving LMCI normalized concentrations. Traceback: {e}")

    try:    
        save(Nifti1Image(mci_voxels, affine=atlas_affine), average_images_dir + 'MCI_norm.nii.gz')
        print("Saved MCI_norm.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving MCI normalized concentrations. Traceback: {e}")
        print(f"Error in saving MCI normalized concentrations. Traceback: {e}")
    try:    
        save(Nifti1Image(emci_voxels, affine=atlas_affine), average_images_dir + 'EMCI_norm.nii.gz')
        print("Saved EMCI_norm.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving EMCI normalized concentrations. Traceback: {e}")
        print(f"Error in saving EMCI normalized concentrations. Traceback: {e}")
    try:    
        save(Nifti1Image(cn_voxels, affine=atlas_affine), average_images_dir + 'CN_norm.nii.gz')
        print("Saved CN_norm.nii.gz brain image")
    except Exception as e:
        logging.error(f"Error in saving CN normalized concentrations. Traceback: {e}")
        print(f"Error in saving CN normalized concentrations. Traceback: {e}")
    
    '''
    logging.info("Z-score normalization of subjects regional concentrations")
    print("Z-score normalization of subjects regional concentrations")
    for pet in tqdm(voxels.keys()):
        try:
            # z-score (region(i) - healthy_mean(i))/healthy_std(i)    
            #voxels[pet] = (voxels[pet] - cn_voxels_mean)/cn_voxels_std
            
            # logistic transformation
            #voxels[pet] = 1 / (1 + np.exp(-voxels[pet])) 
            voxels[pet][np.where(atlas_data==0)] = 0
            
            output_path = pet.replace('.nii.gz', '_avg.nii.gz')          
            save(Nifti1Image(voxels[pet], affine=atlas_affine), output_path)
        except Exception as e:
            logging.error(f"Error in z-score normalization of subjects {pet} regional concentration: {e}")
            print(f"Error in z-score normalization of subjects {pet} regional concentration: {e}")
    '''
    logging.info('*** Done ***')
