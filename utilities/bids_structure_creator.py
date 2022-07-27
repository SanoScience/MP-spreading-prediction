import os
import re
from subprocess import Popen, PIPE, STDOUT
import nibabel as nb
import logging

"""
Create BIDS structure starting from DCOM archive ADNI_2_3_RAW
Current Folder/
    New Folder/
        sub-<ID>/
            ses-baseline/
                dwi/
                    sub-<ID>_ses-1_acq-AP_dwi.[nii, bval, bvec, json]
                anat/
                    sub-<ID>_ses-1_acq-AP_t1.[nii, json]
                pet/
                    sub-<ID>_ses-1_acq-AP_trc-<tracer>_pet.[nii, json]
                    
"""

        
# given the path, it is created if it doesn't exist (mkdir -p creates parent directories if they are missing too)
def check_path(path):
    logging.info("Checking path {}".format(path))
    
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return 

# this method moves the specified @file into the @new_path, giving it a specified @new_name
def move_rename(file, new_path, new_name):
    check_path(new_path)
    os.system("mv {} {}/{}".format(cur_path+sep+file, new_path, new_name))
    return

# short-hand code to get a new name to the file given it's new ID and it's extension
def baptize(id, folder_type):
    return prefix.format(sub_category, id)+"_ses-baseline_acq-AP_{}".format(folder_type)

def build_file_type(name):
    folder_type = ''
    file_type = ''
    date_re = re.compile(r"\d\d\d\d-\d\d-\d\d")
    date = 'date-'+ date_re.search(name).group()
    if re.match(r".*(SPGR|RAGE).*", name):
        folder_type = 'anat'
        file_type = '_anat'
    elif re.match(r".*DTI.*", name):
        folder_type = 'dwi'
        file_type = '_dwi'
    elif re.match(r".*AV-*45.*", name):
        folder_type = 'pet'
        file_type = '_trc-av45_pet'
    elif re.match(r".*FBB.*", name):
        folder_type = 'pet'
        file_type = '_trc-fbb_pet'
    elif re.match(r".*(PIB|pib).*", name):
        folder_type = 'pet'
        file_type = '_trc-pib_pet'
    elif re.match(r".*(Tau|TAU|AV1451).*", name):
        folder_type = 'pet'
        file_type = '_trc-tau_pet'
    file_type = date + file_type
    return folder_type, file_type

def get_new_names(position):
    ''' 
    position can be a subdirectory containing .dcm files or .nii file path
    '''
    split_list = position.split(sep)
    id = split_list[1].split('_')[2]
    folder_type, file_type = build_file_type(position)
    # the absolute new destination path is formalized and checked
    new_path = cur_path + sep + output_folder + sep + prefix.format(sub_category, id) + sep + session + sep + folder_type + sep
    check_path(new_path)
    new_name = baptize(id, file_type)
    return new_path, new_name

logging.basicConfig(level=logging.INFO)

input_folder = input("Insert absolute or relative path of input folder (please be sure there are not spaces): ")
sub_category = input("Insert subjects\' category (i.e. AD, LMCI, EMCI, NC) [default is None]: ")
sub_category = sub_category if len(sub_category)>0 else ''
threads = input("Insert thread number [default 100]: ")
threads = int(threads) if len(threads)>0 else 100
output_folder = input_folder+'_BIDS'
prefix = "sub-{}{}"
session = "ses-baseline"

analyze = r".*\.i$"
nii = r".*.nii$"

sep = os.path.sep # '/'
cur_path = os.path.dirname(os.path.realpath(__file__))

logging.info("Preparing output folder {}".format(output_folder))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    os.system("rm -rf {}".format(output_folder))

processes = []
for subdir, dirs, files in os.walk(input_folder):
    # Checking if the current folder has files inside itself or it's just a "middle directory"
    if files:
        # the file is splitted separating its path from its name. The path is used as dictionary key, to which an ID string is extracted from files inside and bounded as its value.
        # paths[path_to_directory] = ID
        # i.e. /home/luca/Conversion/ADNI_2_3_RAW/003_S_4136/Axial_DTI/2012-09-18_16_13_57.0/I365535/ as key, 4136 as ID (index 4 splitting by '_')
        #logging.info("Directory found: ", subdir)
        
        new_path, new_name = get_new_names(subdir)
        logging.info("Scheduling the conversion of folder {} to {}".format(subdir, new_path))
        
        dicom = True
        for f in files:
            if re.match(analyze, f):
                dicom = False
                try:
                    logging.info(f"found analyze image {subdir}/{f}")
                    img = nb.load(subdir+'/'+f)
                    nb.save(img, f.replace('.i', '.nii'))
                except:
                    logging.info("error, continuing with the next")
            if re.match(nii, f):
                dicom = False
                logging.info(f"{f} is a Nifti file, renaming and moving in new path...")
                command = f"cp {subdir}{sep}{f} {new_path}{new_name}"
        
        if dicom:
            command = f"dcm2niix -f {new_name} -o {new_path} {subdir}"
            
        logging.info(command)
        output = processes.append(Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, executable="/bin/bash"))
    
    if len(processes)%threads == 0:
        for p in processes:
            p.wait()

# All queued conversions are started in parallel
for p in processes:
    p.wait()
    
logging.info("Patients in the input folder:")
os.system(f"ls {input_folder} | wc -l")
logging.info("Original images:")
os.system(f"find {input_folder} -maxdepth 5 -name \'I*\' | wc -l")

logging.info("Patients in output folder:")
os.system(f"ls {output_folder} | wc -l")
logging.info("\'.nii\' files:")
os.system(f"find {output_folder} -name \'*.nii\' | wc -l")

logging.info("DONE")