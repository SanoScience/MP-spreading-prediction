''' Utilities funciton for visualizaiton purposes. '''

import os

import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

#sns.set_theme(style="darkgrid")
#sns.set_palette(['#390099', '#FF0054', '#00A6FB'])

digits = 4

def visualize_NIfTI_data(data, depths, time_idx=None):
    ''' Visualize provided depths from NIfTI data at the specific time. 
    
    Args:
        data (nibabel.nifti1.Nifti1Image): 3D/4D data [width, height, depth, [time]]
        depths (list; len(list)>1): list of depths indexes to visualize
        time_idx (int): index of time; if is None: do not consider time (data is 3D) 
    '''
    data = data.get_fdata()
    
    squeeze = False if len(depths)==1 else True
    fig, axs = plt.subplots(1, len(depths), squeeze=squeeze)
    for i, ax in enumerate(axs):
        ax.imshow(data[:, :, depths[i], time_idx])
        ax.set_title(f'depth number: {depths[i]}')
    plt.suptitle(f'time index: {time_idx}')
    plt.tight_layout()
    plt.show()
    
def visualize_diffusion_timeplot(matrix, timestep, total_time, save_dir=None):
    # TODO: change xticks and labels to time in years
    # plt.xticks(np.arange(0, total_time, step=timestep), labels=np.arange(0, total_time, step=timestep))
    # plt.xlabel('Time [years]' )
    
    plt.imshow(matrix, aspect='auto')
    plt.xlabel('# iterations')
    plt.ylabel('ROIs')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('concentration of amyloid beta', rotation=270)
    plt.title(f'Total time of simulation: {total_time} years')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'diffusion_over_time.png'))
    plt.show() 
    
def save_prediction_plot(baseline, prediction, followup, subject, filepath, error=None, corr_coeff=None):    
    plt.figure(figsize=(18, 10))
    plt.plot(baseline, '-', marker='o', c='#390099', label='baseline', linewidth=1)
    plt.plot(followup, '-', marker='o', c='#00A6FB', label='followup', linewidth=1)
    plt.plot(prediction, '-', marker='o', c='#FF0054', label='predicted', linewidth=4, alpha = 0.4)
        
    '''
    plt.figure(figsize=(25, 10))
    sns.lineplot(data=baseline, label='baseline concentration', marker='o', dashes=False, markers=True, linewidth=1)
    sns.lineplot(data=followup, label='followup concentration', marker='o', dashes=False, markers=True, linewidth=1)
    sns.lineplot(data=prediction, label='predicted followup concentration', marker='o', dashes=True, markers=True, linewidth=3, alpha = 0.5)
    '''
    
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=22)
    plt.xlabel('ROI (166 regions of AAL3 atlas)', fontsize=22)
    plt.ylabel('Concentration of Amyloid-Beta', fontsize=22)
    plt.xlim(-1, len(baseline))
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(len(baseline), step=2), fontsize=10, rotation=35)
    plt.grid(True)
    plt.tight_layout()
    
    if error is not None: plt.title(f'Subject: {subject.split(os.sep)[-2]}, MSE: {round(error, digits)}, PCC: {round(corr_coeff, digits)}', fontsize=22)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    return 

def save_avg_regional_errors(avg_reg_err, avg_reg_err_filename):
    #for i in avg_reg_err:
    #    plt.hist(i)
    
    
    plt.figure(figsize=(20, 10))
    colors = [0, 0, 0]
    for i in range(0, len(avg_reg_err)-1):
        curr_col = avg_reg_err[i+1]/np.max(avg_reg_err)
        colors = [max(curr_col, colors[0]), 0, 0]
        plt.plot([i, i+1], [avg_reg_err[i], avg_reg_err[i+1]], linewidth=3, c=colors)
        colors = [curr_col, 0, 0]
    
    plt.ylabel('Regional error', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-1, len(avg_reg_err))
    plt.xticks(np.arange(len(avg_reg_err), step=2), fontsize=10, rotation=35)
    plt.xlabel('ROI (166 regions of AAL3 atlas)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(avg_reg_err_filename)
    
def visualize_error(error):
    plt.figure(figsize=(15, 5))
    plt.plot(error)
    plt.yscale('log')
    plt.xlabel('# iterations')
    plt.ylabel('reconstruction error')
    plt.show()
