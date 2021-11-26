''' Utilities funciton for visualizaiton purposes. '''

import matplotlib.pyplot as plt

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
    
def visualize_diffusion_timeplot(matrix, save_path=None):
    plt.figure(figsize=(15,3))
    plt.imshow(matrix.T) #, interpolation='nearest'
    plt.xlabel('Iteration' )
    plt.ylabel('ROIs')
    plt.colorbar()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show() 