''' Utilities funciton for visualizaiton purposes. '''

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_palette(['#390099', '#FF0054', '#00A6FB'])

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
    '''
    plt.figure(figsize=(15, 5))
    plt.plot(baseline, '-', marker='o', c='b', label='initial concentration t0')
    plt.plot(prediction, '--', marker='o', c='r', label='predicted concentration t1')
    plt.plot(followup, '-', marker='o', c='g', label='true concentration t1')
    plt.xlabel('ROI (index of brain region based on AAL atlas)')
    plt.ylabel('concentration of misfolded proteins')
    plt.legend(loc='upper right')
    if error is not None: plt.title(f'Subject: {subject} \nError between true and predicted t1: {error:.2f}\nPearson correlation coeff: {corr_coeff:.2f}')
    plt.tight_layout()
    plt.savefig(filepath)
    # After saving the figure, explicitly close it to avoid memory wasting
    plt.close()
    '''
    plt.figure(figsize=(20, 15))
    sns.lineplot(data=baseline, label='baseline concentration', marker='o', dashes=False, markers=True)
    sns.lineplot(data=prediction, label='predicted followup concentration', marker='o', dashes=True, markers=True)
    sns.lineplot(data=followup, label='followup concentration', marker='o', dashes=False, markers=True)
    
    plt.legend(bbox_to_anchor=(1, 1.06), loc='upper right', borderaxespad=0.)
    plt.xlabel('ROI (index of brain region based on AAL3 atlas)')
    plt.ylabel('concentration of misfolded proteins')
    
    if error is not None: plt.title(f'Subject: {subject.split(os.sep)[-2]} \nError between true and predicted t1: {error:.5f}\nPearson correlation coeff: {corr_coeff:.5f}')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return 
    
def visualize_error(error):
    plt.figure(figsize=(15, 5))
    plt.plot(error)
    plt.yscale('log')
    plt.xlabel('# iterations')
    plt.ylabel('reconstruction error')
    plt.show()