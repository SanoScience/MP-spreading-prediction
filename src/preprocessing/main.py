from subprocess import Popen, PIPE, STDOUT
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
from utils.registration import Registration, RegistrationPET
from utils.brain_segmentation import BrainSegmentation
from utils.cerebellum_normalization import CerebellumNormalization
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
import csv
from dipy.segment.mask import median_otsu


def check_path(path):
    #logging.info("Checking path {}".format(path))
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return 

def dispatcher(f, atlas_file, img_type):
    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    #logging.info(f"file: {f}")
    path = f.removesuffix(f.split('/')[-1])
    output_directory = dataset_path + 'derivatives/' + str(re.split(dataset_path, path)[-1])
    intermediate_dir = output_directory + 'intermediate/'
    #output_directory = 'derivatives' + path.removeprefix(path.split('/')[0])
    #output_directory = dataset_path + 'derivatives/' 
    check_path(output_directory)
    check_path(intermediate_dir)

    # From here onward, input files are in absolute path, output is already in the right directory (no need for output directory)
    os.chdir(output_directory)

    # Output name (without path, because the process is already in the output directory)
    name = (f.split('/')[-1]).split('.')[0]

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
            return
        bvals, bvecs = read_bvals_bvecs(name_bval, name_bvec)
        gtab = gradient_table(bvals, bvecs)  
    
    ########################
    ### BRAIN EXTRACTION ###
    ########################
    
    #logging.info(name + " Starting Brain Extraction")
    if img_type == 'anat':
        try:
            be = BET_FSL(name_nii, name, binary_mask=False)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at brain_extraction')
    else:
        img = load(name_nii)
        try:
            be = BrainExtraction(img.get_fdata(), img.affine, img.header , name)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at brain_extraction')
    
    try:
        data, affine, header = be.run()
        bm = be.get_mask()
        name_nii = intermediate_dir + name + '_be.nii.gz'
        name_bm = name + '_bm.nii.gz'
        save(Nifti1Image(data, affine, header), name_nii)
        save(Nifti1Image(bm, affine, header), name_bm)
    except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at brain_extraction (common)')
    #logging.info(name + " Brain Extraction done")

    if img_type == 'dwi':
        #logging.info(name + " Starting LPCA")
        try:
            lpca = Denoising_LPCA(data, affine, header, name, bm)
            data, affine, header = lpca.run(gtab)
            name_nii = intermediate_dir + name + '_lpca.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at LPCA')
        
        #logging.info(name + " LPCA done") 

        #logging.info(name + " Starting Gibbs correction")
        try:
            gib = Gibbs(data, affine, header, name)
            data, affine, header = gib.run()
            name_nii = intermediate_dir + name + '_gibbs.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Gibbs')
        
        #logging.info(name + " Gibbs correction done")
        
        #logging.info(name + " Starting Eddy Current, Motion and Bvec Correction")
        try:
            ec = EddyMotionCorrection(name, name_nii, name_bval, name_bvec, name_json, name_bm)
            data, affine, header = ec.run()
            name_bvec, name_bval = ec.get_bvec_bval()
            gtab = ec.get_BMatrix()
            name_nii = intermediate_dir + name + '_eddy.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Eddy')
        
        #logging.info(name + ' Eddy Current, Motion and Bvec Correction done')
    
    if img_type == 'pet':
        # Motion Correction is needed BEFORE atlas registration
        #logging.info(name + " Starting Motion Correction")
        # only if the image has more than 1 volume, do motion correction
        if len(np.shape(data)) > 3 and np.shape(data)[3] > 1: 
            try:
                mc = MotionCorrection(data, affine, header, name)
                data, affine, header = mc.run()
                name_nii = intermediate_dir + name + '_mc.nii.gz'
                save(Nifti1Image(data, affine, header), name_nii)
            except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at Motion Correction')
            
        #logging.info(name + " Motion Correction done")

        try:
            flat = Flatten(name_nii, name)
            data, affine, header = flat.run()
            name_nii = intermediate_dir + name + '_fl.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Flatten')

        # Atlas registration (FLIRT) needs a concrete file
        #logging.info(name + " Starting Atlas Registration (FLIRT)")
        try:
            atl_regs = RegistrationPET(name_nii, atlas_file, name, img_type)
            data, affine, header = atl_regs.run()
            name_nii = intermediate_dir + name + '_reg.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration')
        #logging.info(name + " Atlas Registration (FLIRT) done")

    else:
        #logging.info(name + " Starting Atlas Registration")
        #atl_regs = Registration(data, affine, header, name)
        #data, affine, header = atl_regs.run(atlas_file, gtab)
        try:
            atl_regs = RegistrationPET(name_nii, atlas_file, name, img_type)
            data, affine, header = atl_regs.run()
            name_nii = intermediate_dir + name + '_reg.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at Registration')
        
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
    
    if img_type == 'anat':
        #logging.info("Starting Brain Segmentation")
        try:
            tissue_class = BrainSegmentation(name_nii, name)
            data, affine, header = tissue_class.run()
            name_nii = intermediate_dir + name + '_segm.nii.gz'
            save(Nifti1Image(data, affine, header), name_nii)
        except Exception as e:
            logging.error(e)
            logging.error(name_nii + ' at BrainSegmentation')
        #logging.info("Brain Segmentation done")
    
    # get the binary mask of dwi after preprocessing
    if img_type == 'dwi':
        try:
            bm = median_otsu(data[:,:,:,0])[0]
            name_bm = name + '_bm.nii.gz'
            save(Nifti1Image(bm, affine, header), name_bm)
        except Exception as e:
                logging.error(e)
                logging.error(name_nii + ' at brain_extraction (final)')
    #logging.info(name + " Brain Extraction done")

    save(Nifti1Image(data, affine, header), name + '.nii.gz')
    #logging.info(f"{name} Preprocessing finished, final output saved as {os.getcwd() + os.sep + name + '.nii'}")

    return 

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
logging.getLogger('nipype.workflow').setLevel(0)
logging.getLogger('nipype.interface').setLevel(0)

if __name__=='__main__':
    # assuming atlas is in the current dir
    atlas_file = os.getcwd() + '/AAL3v1.nii.gz'
    logging.info(f"Using atlas {atlas_file}")

    if len(sys.argv) > 1:
        img_type = sys.argv[1]
    else:
        img_type = input('Provide type of image [dwi/anat/pet/*] or csv filename containing images to process: ')
    #logging.info(f"PreProcessing {img_type} files")

    if img_type.endswith('.csv'):
        files = []
        # the output 'derivative' folder will be created in the same path of passed csv file
        dataset_path = img_type.removesuffix(img_type.split(os.sep)[-1])
        with open(img_type, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='')
            for row in reader:
                files.append(row)
    else:
        if len(sys.argv) > 2:
            dataset_path = sys.argv[2]
        else:
            dataset_path = input('Insert local path of the dataset (enter to look in the current directory): ')
            
        #logging.info(f"Looking for all '.nii' files of type {img_type} in the path {dataset_path} (excluding \'derivatives\' folder)...")
        output = Popen(f"find {dataset_path} ! -path '*derivatives*' ! -wholename '{atlas_file}' -wholename \'*/{img_type}/*.nii\'", shell=True, stdout=PIPE)
        files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

        logging.info("Found {} files".format(len(files)))
        logging.info(files)
            
        # If it starts with '/' it is an absolute path, otherwise make it absolute
        if not dataset_path.startswith('/'):
            if dataset_path != '.':
                dataset_path = os.getcwd() + '/' + dataset_path
            else:
                dataset_path = os.getcwd() + '/'

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

    procs = []
    re_img_type = re.compile(r".*(dwi|pet|anat).*")

    for i in tqdm(range(len(files))):
        
        # this ensure the preprocessing pipeline will execute the right steps for each file (it allows heterogeneity in the list)
        img_type = re_img_type.search(files[i]).group()
        
        #dispatcher(files[i], atlas_file, img_type)
        p = multiprocessing.Process(target=dispatcher, args=(files[i], atlas_file, img_type))
        p.start()
        procs.append(p)
        
        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                # wait for 10 seconds to wait process termination
                p.join(timeout=10)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
                    
        # final chunk could be shorter than num_cores, so it's handled waiting for its completion (join without arguments wait for the end of the process)
        if i == len(files) - 1:
            for p in procs:
                p.join()


    logging.info("Preprocessing done")
