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

def dispatcher(f, atlas_file, img_type, temp_file):
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
        if img_type == 'anat':  
            be = BET_FSL(name_nii, intermediate_dir + name + '_be', binary_mask=False)
        elif img_type == 'dwi':
            img = load(name_nii)
            be = BrainExtraction(img.get_fdata(), img.affine, img.header, name)
    except Exception as e:
        logging.error(e)
        logging.error(name_nii + ' at brain_extraction')
        print(e)
        print(name_nii + ' at brain_extraction')
    
    if img_type == 'anat' or img_type == 'dwi':
        logging.info(f"{name_nii} starting brain extraction")
        try:
            data, affine, header = be.run()
            bm_data = be.get_mask()
            del be               
            
            logging.info("1")
            
            ### Crop images to save space...
            img = crop_img(Nifti1Image(data, affine, header))
            data, affine, header = img.get_fdata(), img.affine, img.header 
            del img
            
            logging.info("2")
            
            # Binary mask has to be mandatorily saved for Eddy
            name_bm = name + '_bm.nii.gz'
            save(Nifti1Image(bm_data, affine, header), name_bm)
            bm_img = crop_img(name_bm) 
            save(bm_img, name_bm)
            bm_data = bm_img.get_fdata()
            del bm_img
            
            logging.info("3")
            
            if temp_file:
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
            
            if temp_file:
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
            
            if temp_file:
                name_nii = intermediate_dir + name + '_gibbs.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Gibbs')
            print(e)
            print(name_nii + ' at Gibbs')
            
        logging.info(f"{name_nii} starting Eddy")
        try:
            ec = EddyMotionCorrection(name, name_nii, name_bval, name_bvec, name_json, name_bm)
            data, affine, header = ec.run()
            name_bvec, name_bval = ec.get_bvec_bval()
            gtab = ec.get_BMatrix()
            del ec
            
            if temp_file:
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
        logging.info(f"{name_nii} starting Brain Extraction (PET)")
        try:
            be = BrainExtraction(data, affine, header , name)
            data, affine, header = be.run()
            bm_data = be.get_mask()
            del be
            
            img = crop_img(Nifti1Image(data, affine, header))
            data, affine, header = img.get_fdata(), img.affine, img.header
            del img
            # cropped binary mask has to be saved before registration
            save(crop_img(Nifti1Image(bm_data, affine, header)), name_bm)
            
            if temp_file:
                name_nii = intermediate_dir + name + '_be.nii.gz'
                name_bm = name + '_bm.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)         
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Brain Extraction (PET)')
            print(e)
            print(name_nii + ' at Brain Extraction (PET)')
            
        img = crop_img(name_nii)
        data, affine, header = img.get_fdata(), img.affine, img.header
        del img
        # Motion Correction is needed BEFORE atlas registration (only if the image has more than 1 volume)
        if len(np.shape(data)) > 3 and np.shape(data)[3] > 1: 
            logging.info(f"{name_nii} starting Motion Correction")
            try:
                mc = MotionCorrection(data, affine, header, name)
                data, affine, header = mc.run()
                del mc
                if temp_file:
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
                if temp_file:
                    name_nii = intermediate_dir + name + '_fl.nii.gz'
                    save(Nifti1Image(data, affine, header), name_nii)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Flatten')
                print(e)
                print(name_nii + ' at Flatten')

        try: 
            # NOTE: binary mask is obtained on the first slice of PET, which is not moved by motion correction nor by flattening
            logging.info(f"{name_nii} starting Registration of binary mask (PET)")
            bm_reg = Registration(name_bm, atlas_file, intermediate_dir+name, 'mask')
            bm_data, bm_affine, bm_header = bm_reg.run()
            del bm_reg
            # Binary mask is always saved (it is not an intermediate output)
            save(Nifti1Image(bm_data, bm_affine, bm_header), name_bm)
            
            logging.info(f"{name_nii} starting Registration (PET)")
            pet_reg = Registration(name_nii, atlas_file, intermediate_dir+name, img_type)
            data, affine, header = pet_reg.run()
            del pet_reg
            if temp_file:
                name_nii = intermediate_dir + name + '_reg.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration (PET)')
            print(e)
            print(name_nii + ' at Registration (PET)')
    
        gc.collect()
    ###################################
    ### REGISTRATION (DWI and ANAT) ###
    ###################################
    else:
        logging.info(f"{name_nii} starting Registration")
        try:
            atl_regs = Registration(name_nii, atlas_file, name, img_type)
            data, affine, header = atl_regs.run()
            del atl_regs
            if temp_file:
                name_nii = intermediate_dir + name + '_reg.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration')
            print(e)
            print(name_nii + ' at Registration')
        
    '''
    NOTE: DEPRECATED
    if img_type == 'pet':    
        #logging.info(name + " Starting Cerebellum Normalization")
        try:
            ce = CerebellumNormalization(name_nii, atlas_file, name)
            data, affine, header = ce.run()
            name_nii = intermediate_dir + name + '_norm.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at CerebellumNormalization')
        #logging.info(name + " Cerebellum Normalization done")
    '''
    
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
            if temp_file:
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
        output = Popen(f"find {dataset_path} ! -path '*derivatives*' ! -wholename '{atlas_file}' -name \'*{img_type}.nii\'", shell=True, stdout=PIPE)
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
        
    if len(sys.argv) > 4:
        temp_file = sys.argv[4]
    else:
        try: 
            temp_file = input("Do you want to produce intermediate outputs [y/N]? ")
        except:
            temp_file = ''
    temp_file = True if temp_file == 'y' or temp_file == 'Y' else False        

    # TODO
    # skip_temporary = True if input('Do you want to skip already processed temp files? [Y/n]') != 'n' else False

    logging.info('******************************************')
    logging.info(f"Atlas: {atlas_file}")
    logging.info(f"Images list or type: {txt_list}")
    logging.info(f"Dataset path provided: {dataset_path}")
    logging.info(f"Cores: {num_cores}")
    logging.info(f"Intermediate files {'enabled' if temp_file else 'disabled'}")
    logging.info(f"Number of images to process: {len(files)}")
    logging.info("List of images to process: ")
    logging.info(files)
    logging.info('******************************************')

    procs = []
    re_img_type = re.compile(r"(dwi|pet|anat)")

    for i in tqdm(range(len(files)), file=sys.stdout):
        
        # this ensure the preprocessing pipeline will execute the right steps for each file (it allows heterogeneity in the list)
        img_type = re_img_type.search(files[i]).group()
        
        p = multiprocessing.Process(target=dispatcher, args=(files[i], atlas_file, img_type, temp_file))
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