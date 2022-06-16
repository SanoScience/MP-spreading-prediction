""" 
    SYNOPSYS
    python3 main.py <img_type> <dataset_path> <cores> <atlas_file> 
    
    NOTES:
    register anat to atlas = AAL3v1.nii.gz
    register dwi to preprocessed anatomical: atlas_file = 'anat'
    register pet to skull atlas = MNI152_T1_1mm.nii.gz
"""

from subprocess import Popen, PIPE, STDOUT
from nibabel import load, save, Nifti1Image
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import os
import re
import sys
from utils.brain_extraction import BrainExtraction, BET_FSL
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
from threading import Thread, active_count
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from datetime import datetime
from dipy.segment.mask import median_otsu
from nilearn.image import crop_img
import gc
from glob import glob


def check_path(path):
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return 

class Preprocess(Thread):
    def __init__(self, f, img_type):
        Thread.__init__(self)
        self.f = f 
        self.img_type = img_type

    def run(self):
        self.f = self.f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
        path = os.getcwd() + os.sep + self.f.removesuffix(self.f.split('/')[-1])
        output_directory = dataset_path + 'derivatives/' + str(re.split(dataset_path, path)[-1])
        check_path(output_directory)
        # NOTE: intermediate_dir must exist because some modules use it to store their output (i.e. FLIRT)
        intermediate_dir = output_directory + 'intermediate/'
        check_path(intermediate_dir)

        # From here onward, input files are in absolute path, output is already in the right directory (no need for output directory)
        os.chdir(output_directory)

        # Output name (without path, because the process is already in the output directory)
        name = (self.f.split('/')[-1]).split('.')[0]
        
        del self.f

        # Inputs
        name_nii = path + name + '.nii'
        name_json = path + name + '.json'
        name_bm = name + '_mask.nii.gz'
        
        # NOTE: Optionally, you can use registered anatomical (if any) to register the dwi
        if self.img_type != 'anat' and atlas_file == 'anat':
            try:
                # NOTE: the search is always directed under the 'ses-baseline' folder, to be sure to have always an anatomical
                atlas_file = glob(output_directory + '..' + os.sep + '..' + os.sep + 'ses-baseline' + os.sep + 'anat' + os.sep + 'sub*_anat.nii.gz')[0]
                logging.info(f"Found image {atlas_file} to register {name_nii} to")
                print(f"Found image {atlas_file} to register {name_nii} to")
            except Exception as e:
                logging.error(e)
                logging.info(f"Using standard atlas for registration of {name_nii}")
                atlas_file = 'AAL3v1.nii.gz'
                print(e)
                print(f"Using standard atlas for registration of {name_nii}")

        os.system(f"cp {name_json} {name+'.json'}") # copy json in the output folder
        gtab = None # if gtab is 'None' (example, for anat and pet images) don't use it in registration
        if self.img_type == 'dwi':
            try:
                name_bval = path + name + ".bval"
                name_bvec = path + name + ".bvec"
                # NOTE: bval and bvec files will be overwritten by Eddy and by registration
                os.system(f"cp {name_bval} {name+'.bval'}")
                os.system(f"cp {name_bvec} {name+'.bvec'}")
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
        
        if self.img_type == 'anat' or self.img_type == 'dwi': 
            try: 
                be = BET_FSL(name_nii, intermediate_dir + name + '_be', self.img_type)
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

        if self.img_type == 'dwi':
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
        
        if self.img_type == 'pet':
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
            
            img = crop_img(name_nii)
            data, affine, header = img.get_fdata(), img.affine, img.header
            del img        
            
            try:             
                logging.info(f"{name_nii} starting Registration (PET)")
                pet_reg = Registration(name_nii, atlas_file, intermediate_dir, name, self.img_type)
                data, affine, header = pet_reg.run()
                del pet_reg
                
                name_nii = intermediate_dir + name + '_reg.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Registration (PET)')
                print(e)
                print(name_nii + ' at Registration (PET)')
            
                #create binary mask (at this point the PET is 3D!)
                _, bm = median_otsu(data)
                name_bm = name + '_mask.nii.gz'
                # binary mask is always saved
                save(Nifti1Image(bm, affine, header), name_bm)

        gc.collect()

        ###################################
        ### REGISTRATION (DWI and ANAT) ###
        ###################################
        if self.img_type == 'dwi' or self.img_type == 'anat':
            logging.info(f"{name_nii} starting Registration")
            try:
                atl_regs = Registration(name_nii, atlas_file, intermediate_dir, name, self.img_type)
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
        
        if self.img_type == 'anat':
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
        
        # get the binary mask of dwi after preprocessing (without cropping anything)
        if self.img_type == 'dwi':
            logging.info(f"{name_nii} starting Final Masking (DWI)")
            try:
                _, bm = median_otsu(data[:,:,:,0])
                name_bm = name + '_mask.nii.gz'
                # binary mask is always saved
                save(Nifti1Image(bm, affine, header), name_bm)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Final Masking (DWI)')
                print(e)
                print(name_nii + ' at Final Masking (DWI)')
                
        # final save is mandatory
        save(Nifti1Image(data, affine, header), name + '.nii.gz')
        logging.info(f"{name + '.nii.gz'} final image saved")

        return 

start_time = datetime.today()
logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, force = True, filename = f"trace_{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.log")

if __name__=='__main__':
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
        output = Popen(f"find {dataset_path} ! -path \'*derivatives*\' -name \'*{img_type}.nii\'", shell=True, stdout=PIPE)
        files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
        
    # If it starts with '/' it is an absolute path, otherwise make it absolute
    if not dataset_path.startswith('/'): 
        dataset_path = os.getcwd() + os.sep + dataset_path if dataset_path != '.' else os.getcwd() + os.sep
    os.chdir(dataset_path)

    if len(sys.argv) > 3:
        if int(sys.argv[3]) == -1:
            num_cores = cpu_count()
        else:
            num_cores = int(sys.argv[3])
    else:
        num_cores = input("Insert the number of cores you want to use (default, 4): ")
        num_cores = int(num_cores) if len(num_cores) > 0 else 4
        
    if len(sys.argv) > 4:
        # 'anat' to coregister to the preprocessed anatomical image (if it exists)
        atlas_file = sys.argv[4]
    else:
        atlas_file = input("Insert the path to the atlas you want to use [default, register to the anatomical image]: ")
        if len(atlas_file) == 0:
            atlas_file = 'anat'
    if (not atlas_file == 'anat') and (not atlas_file.startswith(os.sep)):
        atlas_file = os.getcwd() + os.sep + atlas_file
        #os.getcwd() + '/AAL3v1.nii.gz'

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

    for i in tqdm(range(len(files)), file=sys.stdout):
        
        # this ensure the preprocessing pipeline will execute the right steps for each file (it allows heterogeneity in the list)
        # NOTE: even if this overwrites "img_type", the images belonging to the specified type have been already loaded
        img_type = re_img_type.search(files[i]).group()

        Preprocess(files[i], img_type).start()            
        logging.info(f"Image {files[i]} ({img_type}) queued")
        while active_count() > num_cores + 1:
            pass
            
    while active_count() > 2:
        pass

    total_time = (datetime.today() - start_time).seconds
    print(f"Preprocessing done in {total_time} seconds")
    
    logging.info(f"Preprocessing done in {total_time} seconds")
    logging.info('******************************************')
