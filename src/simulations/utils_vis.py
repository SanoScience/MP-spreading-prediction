''' Utilities funciton for visualizaiton purposes. '''

import os

import matplotlib.pyplot as plt
import numpy as np

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
    plt.colorbar()
    plt.title(f'Total time of simulation: {total_time} years')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'diffusion_over_time.png'))
    plt.show() 
    
def visualize_terminal_state_comparison(input_vec, output_vec, target_vec, rmse, save_dir=None):
    plt.figure(figsize=(15, 5))
    plt.plot(input_vec, '--', marker='o', c='b', label='initial concentration t0')
    plt.plot(output_vec, '--', marker='o', c='r', label='predicted concentration t1')
    plt.plot(target_vec, '--', marker='o', c='g', label='true concentration t1')
    plt.xlabel('ROI (index of brain region based on AAL atlas)')
    plt.ylabel('concentration of misfolded proteins')
    plt.legend(loc='upper right')
    plt.title(f'RMSE between true and predicted t1: {rmse:.2f}')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'concentration_comparison.png'))
    plt.show()