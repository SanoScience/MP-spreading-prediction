import logging
import os
from glob import glob

import nibabel
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def load_matrix(path):
    data = np.genfromtxt(path, delimiter=",")
    return data

def load_atlas(path):
    data = nibabel.load(path).get_fdata()
    return data

def prepare_atlas_with_concentrations(atlas, concentration):
    regions = np.unique(atlas).astype('int')
    
    # insert MP concentration = 0 for background
    concentration = list(concentration)
    concentration.insert(0, 0)
    
    for region_idx in regions:
        atlas = np.where(atlas == region_idx, concentration[region_idx], atlas)
        
    # drop negative concentrations (especially in predictions)
    atlas = np.where(atlas < 0, 0, atlas)
    
    return atlas

def visualize_region_concentration(brain, slice_num=50, title_label=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
    im1 = ax1.imshow(np.rot90(brain[slice_num, :, :]), cmap='Reds', aspect="auto")
    ax1.axis('off')
    ax2.imshow(np.rot90(brain[:, slice_num, :]), cmap='Reds', aspect="auto")
    ax2.axis('off')
    ax3.imshow(np.rot90(brain[:, :, slice_num]), cmap='Reds', aspect="auto")
    ax3.axis('off')

    fig.colorbar(im1)
    plt.suptitle(f'MP concentration in brain regions: {title_label}')
    plt.tight_layout()
    plt.show()  
    
def plotly_vis(brain):
    brain = np.rot90(brain)
    fig = px.imshow(brain[:, :, :].T, animation_frame=0, 
                    color_continuous_scale='Reds',
                    zmin=0)
    fig.show()
    
def main():
    output_dir = '../../results'                                                
    dataset_dir = '../../data/ADNI/derivatives/'
    atlas_path = '../../data/atlas/aal.nii.gz'
    
    patients = ['sub-AD4009']
    for subject in patients:
        logging.info(f'\Visualization for subject: {subject}')
        
        # load predicted t1 concentration
        pred_concentrations_path = os.path.join(output_dir, subject, 'concentration_pred_MAR.csv')
        t1_concentration_pred = load_matrix(pred_concentrations_path) 
        
        # load true t1 concentration
        true_concentrations_paths = glob(os.path.join(dataset_dir, subject, 'ses*', 'pet', '*.csv'))                  
        t1_concentration_path = [path for path in true_concentrations_paths if 'followup' in path][0]            
        t1_concentration = load_matrix(t1_concentration_path)
       
        atlas = load_atlas(atlas_path)
        
        brain_with_MP_true = prepare_atlas_with_concentrations(atlas, t1_concentration)
        brain_with_MP_pred = prepare_atlas_with_concentrations(atlas, t1_concentration_pred)
        
        visualize_region_concentration(brain_with_MP_true, title_label='t1 true')
        visualize_region_concentration(brain_with_MP_pred, title_label='t1 pred')
        
        plotly_vis(brain_with_MP_true)

if __name__ == '__main__':
    main()