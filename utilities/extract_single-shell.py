from nibabel import load, save, Nifti1Image
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import sys
from dipy.io.gradients import read_bvals_bvecs
from tqdm import tqdm

if len(sys.argv) > 1:
        multi_shell_list = sys.argv[1]
else:
    print("Please pass a txt containing multishell dwi images (one line per file)")
    quit()

files = open(multi_shell_list, "r").readlines()

for f in tqdm(files):
    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '').removesuffix("\n")
    print(f"Extracting single shell for file {f}")
    
    # Read data
    bval_file = f.replace('.nii', '.bval')
    bvec_file = f.replace('.nii', '.bvec')
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

    img = load(f)
    mu_sh_data = np.array(img.get_fdata())
    new_header = img.header
    print(f"Original shape of data: {mu_sh_data.shape}")
    
    # Count number of single shell bvalues (after the first zeroes, don't count them anymore)
    first_zero = True
    counter = 0
    for b in bvals:
        if (first_zero and b == 0.0) or b == 1000.0: counter +=1
        if b == 1000.0: first_zero = False
    print(f"{counter} single shell values")
            
    # cast the empty np array with the number of expected single shell slices
    si_sh_data = np.empty((mu_sh_data.shape[0], mu_sh_data.shape[1], mu_sh_data.shape[2], counter))
    si_sh_bvec = np.empty((counter, 3))
    
    # prepare new bvac and bvec files (overwrite old multishell version)
    new_bval = open(bval_file, 'w')
    first_zero = True
    counter_values = 0 
    for b in range(len(bvals)):
        if (first_zero and bvals[b] == 0.0) or bvals[b] == 1000.0:
            si_sh_data [...,counter_values] = mu_sh_data[...,b]
            new_bval.write(str(int(bvals[b])) + ' ')
            si_sh_bvec[counter_values, 0] = bvecs[b, 0]
            si_sh_bvec[counter_values, 1] = bvecs[b, 1]
            si_sh_bvec[counter_values, 2] = bvecs[b, 2]
            counter_values += 1
            if bvals[b] == 1000.0: first_zero = False
                
    new_bval.write('\n')
    new_bval.close()
    
    new_bvec = open(bvec_file, 'w')
    for i in range(3):
        for j in si_sh_bvec:
            new_bvec.write(str(j[i]) + ' ')
        new_bvec.write('\n')
    new_bvec.close()
    
    print(f"New shape of data: {np.shape(si_sh_data)}")
    # modify the header with the new number of volumes
    new_header['dim'][4] = np.shape(si_sh_data)[3]

    # overwrite single shell image
    save(Nifti1Image(si_sh_data, img.affine, new_header), f)
    print(f"Image {f} extracted and auxiliary files (.bval and .bvec) overwritten")