from subprocess import Popen, PIPE, STDOUT
from tracemalloc import start
from nibabel import load, save, Nifti1Image
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import re
import sys
from utils.brain_extraction import BrainExtraction, BET_FSL
from utils.brain_extraction import BrainExtraction
from utils.cerebellum_normalization import CerebellumNormalization
from utils.denoising import Denoising_LPCA
from utils.eddy_correction import EddyMotionCorrection
from utils.flatten import Flatten
from utils.gibbs import Gibbs
from utils.motion_correction import MotionCorrection
from utils.registration import Registration, Registration
from utils.brain_segmentation import BrainSegmentation
import logging
import logging.handlers
import multiprocessing
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dipy.segment.mask import median_otsu
from nilearn.image import crop_img
import gc

def check_path(path):
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return 

def dispatcher(f, atlas_file, img_type):
    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    path = os.getcwd() + os.sep + f.removesuffix(f.split('/')[-1])
    output_directory = dataset_path + 'derivatives/' + str(re.split(dataset_path, path)[-1])
    check_path(output_directory)
    # NOTE: intermediate_dir must exist because some modules use it to store their output (i.e. FLIRT)
    intermediate_dir = output_directory + 'intermediate/'
    check_path(intermediate_dir)

    # From here onward, input files are in absolute path, output is already in the right directory (no need for output directory)
    os.chdir(output_directory)

    # Output name (without path, because the process is already in the output directory)
    name = (f.split('/')[-1]).split('.')[0]
    
    del f

    # Inputs
    name_nii = path + name + '.nii.gz'
    name_json = path + name + '.json'
    name_bm = name + '_bm.nii.gz'
    
    os.system(f"cp {name_json} {name+'.json'}") # copy json in the output folder
    gtab = None # if gtab is 'None' (example, for anat and pet images) don't use it in registration
    if img_type == 'dwi':
        try:
            name_bval = path + name + ".bval"
            name_bvec = path + name + ".bvec"
        except Exception as e:
            logging.error(e)
            print(e)
            return
        bvals, bvecs = read_bvals_bvecs(name_bval, name_bvec)
        gtab = gradient_table(bvals, bvecs)  
    
    del path 
    del output_directory
    gc.collect()    
    
    ########################
    ### BRAIN EXTRACTION ###
    ########################
    
    if img_type == 'pet':
        try:
            logging.info(f"{name_nii} starting Cerebellum Normalization")
            ce = CerebellumNormalization(name_nii, atlas_file, intermediate_dir, name)
            data, affine, header = ce.run()
            name_nii = intermediate_dir + name + '_norm.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at CerebellumNormalization')
            print(e)
            print(name_nii + ' at CerebellumNormalization')
    
    gc.collect()
    #############################
    ### FINAL DWI BINARY MASK ###
    #############################
    
    # final save is mandatory
    save(Nifti1Image(data, affine, header), name + '.nii.gz')
    logging.info(f"{name + '.nii.gz'} final image saved")

    return 

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

if __name__=='__main__':
    # assuming atlas is in the current dir
    atlas_file = os.getcwd() + '/AAL3v1.nii.gz'

    if len(sys.argv) > 1:
        txt_list = sys.argv[1]
    else:
        try:
            txt_list = input('Provide txt filename containing images to process (1 line per file) [optional, Enter to process all images, or enter [dwi,anat,pet] to process a specific category]: ')
        except:
            txt_list = ''
    #logging.info(f"PreProcessing {img_type} files")
        
    if len(sys.argv) > 2:
        dataset_path = sys.argv[2]
    else:
        dataset_path = input('Insert local path of the dataset (enter to look in the current directory): ')
        if len(dataset_path) == 0: dataset_path = '.'
        
    if txt_list.endswith('.txt'):
        f = open(txt_list, "r")
        files = f.readlines()
    else:
        if len(txt_list)>0:
            img_type = txt_list
        else:
            img_type = '*'
        #logging.info(f"Looking for all '.nii' files in the path {dataset_path} (excluding \'derivatives\' folder)...")
        output = Popen(f"find {dataset_path} ! -path \'*derivatives*\' ! -name \'{atlas_file.split(os.sep)[-1]}\' -name \'*{img_type}.nii.gz\'", shell=True, stdout=PIPE)
        files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
        
    # If it starts with '/' it is an absolute path, otherwise make it absolute
    if not dataset_path.startswith('/'): dataset_path = os.sep.join(os.getcwd(), dataset_path) if dataset_path != '.' else os.getcwd() + os.sep
    os.chdir(dataset_path)

    if len(sys.argv) > 3:
        if int(sys.argv[3]) == -1:
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = int(sys.argv[3])
    else:
        num_cores = input("Insert the number of cores you want to use (default, 4): ")
        num_cores = int(num_cores) if len(num_cores) > 0 else 4

    # TODO
    # skip_temporary = True if input('Do you want to skip already processed temp files? [Y/n]') != 'n' else False

    logging.info('******************************************')
    logging.info(f"Atlas: {atlas_file}")
    logging.info(f"Images list or type: {txt_list}")
    logging.info(f"Dataset path provided: {dataset_path}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Number of images to process: {len(files)}")
    logging.info("List of images to process: ")
    logging.info(files)
    logging.info('******************************************')

    procs = []
    re_img_type = re.compile(r"(dwi|pet|anat)")

    for i in tqdm(range(len(files)), file=sys.stdout):
        
        # this ensure the preprocessing pipeline will execute the right steps for each file (it allows heterogeneity in the list)
        img_type = re_img_type.search(files[i]).group()
        
        p = multiprocessing.Process(target=dispatcher, args=(files[i], atlas_file, img_type))
        p.start()
        procs.append(p)
        logging.info(f"Image {files[i]} queued")
        
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                # wait for 10 seconds to wait process termination
                p.join(timeout=2)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
                    del p 
                    gc.collect()
                    
        # final chunk could be shorter than num_cores, so it's handled waiting for its completion (join without arguments wait for the end of the process)
        if i == len(files) - 1:
            for p in procs:
                p.join()

    total_time = (datetime.today() - start_time).seconds
    print(f"Preprocessing done in {total_time} seconds")
    
    logging.info(f"Preprocessing done in {total_time} seconds")
    logging.info('******************************************')