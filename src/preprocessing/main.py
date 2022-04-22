import fractions
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
    name_nii = path + name + '.nii'
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
    
    try:
        if img_type == 'anat' or img_type == 'dwi':  
            be = BET_FSL(name_nii, intermediate_dir + name + '_be')
    except Exception as e:
        logging.error(e)
        logging.error(name_nii + ' at brain_extraction')
        print(e)
        print(name_nii + ' at brain_extraction')
        
        logging.info(f"{name_nii} starting brain extraction")
        try:
            data, affine, header = be.run()
            bm_data = be.get_mask()
            del be               
            
            ### Crop images to save space...
            img = crop_img(Nifti1Image(data, affine, header))
            data, affine, header = img.get_fdata(), img.affine, img.header 
            del img
            
            # Binary mask has to be mandatorily saved for Eddy
            save(Nifti1Image(bm_data, affine, header), name_bm)
            bm_img = crop_img(name_bm) 
            save(bm_img, name_bm)
            bm_data = bm_img.get_fdata()
            del bm_img
            
            name_nii = intermediate_dir + name + '_be.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at brain_extraction (common)')
            print(e)
            print(name_nii + ' at brain_extraction (common}')
           
    gc.collect()
    #######################
    ### DENOISING (DWI) ###
    #######################

    if img_type == 'dwi':
        logging.info(f"{name_nii} starting LPCA")
        try:
            lpca = Denoising_LPCA(data, affine, header, name, bm_data)
            data, affine, header = lpca.run(gtab)
            del lpca
            del gtab
            
            name_nii = intermediate_dir + name + '_lpca.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at LPCA')
            print(e)
            print(name_nii + ' at LPCA')
        
        logging.info(f"{name_nii} starting Gibbs")
        try:
            gib = Gibbs(data, affine, header, name)
            data, affine, header = gib.run()
            del gib 
            
            name_nii = intermediate_dir + name + '_gibbs.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Gibbs')
            print(e)
            print(name_nii + ' at Gibbs')
            
        logging.info(f"{name_nii} starting Eddy")
        try:
            ec = EddyMotionCorrection(name, name_nii, name_bval, name_bvec, name_json, name_bm, intermediate_dir)
            data, affine, header = ec.run()
            name_bvec, name_bval = ec.get_bvec_bval()
            gtab = ec.get_BMatrix()
            del ec
            
            name_nii = intermediate_dir + name + '_eddy.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Eddy')
            print(e)
            print(name_nii + ' at Eddy')

    gc.collect()
    ######################################
    ### PET PREPARATION & REGISTRATION ###
    ######################################
    
    if img_type == 'pet':
        img = load(name_nii)
        data, affine, header = img.get_fdata(), img.affine, img.header
        
        # Motion Correction is needed BEFORE brain extraction, because output will be truncated to one single volume
        if len(np.shape(data)) > 3 and np.shape(data)[3] > 1: 
            logging.info(f"{name_nii} starting Motion Correction")
            try:
                mc = MotionCorrection(data, affine, header, name)
                data, affine, header = mc.run()
                del mc
                name_nii = intermediate_dir + name + '_mc.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Motion Correction')
                print(e)
                print(name_nii + ' at Motion Correction')

            logging.info(f"{name_nii} starting Flatten")
            try:
                flat = Flatten(name_nii, name)
                data, affine, header = flat.run()
                del flat
                
                name_nii = intermediate_dir + name + '_fl.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Flatten')
                print(e)
                print(name_nii + ' at Flatten')
                
        logging.info(f"{name_nii} starting Brain Extraction (PET)")
        try:
            be = BET_FSL(name_nii, intermediate_dir + name + '_be')
            data, affine, header = be.run(frac=0.1, vertical_gradient=-0.3)
            bm_data = be.get_mask()
            del be
            
            name_nii = intermediate_dir + name + '_be.nii.gz'
            name_bm = name + '_mask.nii.gz'
            
            ### Crop images to save space...
            img = crop_img(Nifti1Image(data, affine, header))
            data, affine, header = img.get_fdata(), img.affine, img.header 
            save(img, name_nii)    
            del img
            
            save(Nifti1Image(bm_data, affine, header), name_bm)
            bm_img = crop_img(name_bm) 
            save(bm_img, name_bm)
            bm_data = bm_img.get_fdata()
            del bm_img
                 
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Brain Extraction (PET)')
            print(e)
            print(name_nii + ' at Brain Extraction (PET)')
            
        img = crop_img(name_nii)
        data, affine, header = img.get_fdata(), img.affine, img.header
        del img        

        try: 
            # NOTE: binary mask is obtained on the first slice of PET, which is not moved by motion correction nor by flattening
            logging.info(f"{name_nii} starting Registration of binary mask (PET)")
            bm_reg = Registration(name_bm, atlas_file, intermediate_dir, name, 'mask')
            bm_data, bm_affine, bm_header = bm_reg.run()
            del bm_reg
            # Binary mask is always saved (it is not an intermediate output)
            save(Nifti1Image(bm_data, bm_affine, bm_header), name_bm)
            
            logging.info(f"{name_nii} starting Registration (PET)")
            pet_reg = Registration(name_nii, atlas_file, intermediate_dir, name, img_type)
            data, affine, header = pet_reg.run()
            del pet_reg
            
            name_nii = intermediate_dir + name + '_reg.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration (PET)')
            print(e)
            print(name_nii + ' at Registration (PET)')
            
        # registration is required prior Cerebellum Normalization
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
    ###################################
    ### REGISTRATION (DWI and ANAT) ###
    ###################################
    else:
        logging.info(f"{name_nii} starting Registration")
        try:
            atl_regs = Registration(name_nii, atlas_file, intermediate_dir, name, img_type)
            data, affine, header = atl_regs.run()
            del atl_regs
            name_nii = intermediate_dir + name + '_reg.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration')
            print(e)
            print(name_nii + ' at Registration')
        
    gc.collect()
    #########################
    ### ANAT SEGMENTATION ###
    #########################
    
    if img_type == 'anat':
        logging.info(f"{name_nii} starting Brain Segmentation (ANAT)")
        try:
            tissue_class = BrainSegmentation(name_nii, name)
            data, affine, header = tissue_class.run()
            del tissue_class
            name_nii = intermediate_dir + name + '_segm.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Brain Segmentation (ANAT)')
            print(e)
            print(name_nii + ' at Brain Segmentation (ANAT)')
    
    gc.collect()
    #############################
    ### FINAL DWI BINARY MASK ###
    #############################
    
    # get the binary mask of dwi after preprocessing (simple sweep of median_otsu, without particular settings)
    if img_type == 'dwi':
        logging.info(f"{name_nii} starting Final Brain Extraction (DWI)")
        try:
            bm = median_otsu(data[:,:,:,0])[0]
            name_bm = name + '_bm.nii.gz'
            # binary mask is always saved
            save(Nifti1Image(bm, affine, header), name_bm)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Final Brain Extraction (DWI)')
            print(e)
            print(name_nii + ' at Final Brain Extraction (DWI)')
            
    # final save is mandatory
    save(Nifti1Image(data, affine, header), name + '.nii.gz')
    logging.info(f"{name + '.nii.gz'} final image saved")

    return 

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

if __name__=='__main__':
    # assuming atlas is in the current dir
    atlas_file = os.getcwd() + '/AAL3v1.nii.gz'
    re_img_type = re.compile(r"(dwi|pet|anat)")

    if len(sys.argv) > 1:
        img_type = sys.argv[1]
    else:
        try:
            img_type = input('Provide txt filename containing images to process (1 line per file) [optional, Enter to process all images, or enter [dwi,anat,pet] to process a specific category]: ')
        except:
            img_type = ''
    #logging.info(f"PreProcessing {img_type} files")
        
    if len(sys.argv) > 2:
        dataset_path = sys.argv[2]
    else:
        dataset_path = input('Insert local path of the dataset (enter to look in the current directory): ')
        if len(dataset_path) == 0: dataset_path = '.'
        
    if img_type.endswith('.txt'):
        f = open(img_type, "r")
        files = f.readlines()
    else:
        if not re.match(re_img_type, img_type):
            img_type = '*'
        #logging.info(f"Looking for all '.nii' files in the path {dataset_path} (excluding \'derivatives\' folder)...")
        output = Popen(f"find {dataset_path} ! -path \'*derivatives*\' ! -name \'{atlas_file.split(os.sep)[-1]}\' -name \'*{img_type}.nii\'", shell=True, stdout=PIPE)
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
    logging.info(f"Images list or type: {img_type}")
    logging.info(f"Dataset path provided: {dataset_path}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Number of images to process: {len(files)}")
    logging.info("List of images to process: ")
    logging.info(files)
    logging.info('******************************************')

    procs = []

    for i in tqdm(range(len(files)), file=sys.stdout):
        
        # this ensure the preprocessing pipeline will execute the right steps for each file (it allows heterogeneity in the list)
        # NOTE: even if this overwrites "img_type", the images belonging to the specified type have been already loaded
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