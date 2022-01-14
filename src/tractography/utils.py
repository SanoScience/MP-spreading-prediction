from distutils.command.config import config
import os
from multiprocessing import Process
import logging
from tqdm import tqdm
import re
from glob import glob

def task_completion_info(sound_duration=1, sound_freq=440):
    print('Process completed.')
    os.system(f'play -nq -t alsa synth {sound_duration} sine {sound_freq}')  
    
def parallelize(dwi_files, num_cores, func, config, general_dir):
    ''' Run function on multiple subjects in parallel. '''
    
    logging.info(f'Parallelization started ({num_cores} cores).')
    
    procs = []
    # instantiating process with arguments
    for i in tqdm(range(len(dwi_files))):
        stem_t1 = glob(general_dir + config['paths']['dataset_dir'] + os.sep + config['paths']['subject'] + os.sep + 'ses-*' + os.sep + 'anat' + os.sep + '*_t1.nii')[0].split('.')[0]
        """
        sub_path = dwi_files[i].removesuffix(dwi_files[i].split(os.sep)[-2] + os.sep + dwi_files[i].split(os.sep)[-1])
        for path, dirs, files in os.walk( sub_path + 'anat'):
            for f in files:
                if re.match(r".*t1.nii", f):
                    # all t1s have the same date
                    stem_t1 = os.path.join(path, f.split('.')[0])
                    break
        """

        proc = Process(target=func, args=(dwi_files[i].split('.')[0], stem_t1, config))
        procs.append(proc)
        proc.start()

        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                p.join(timeout=60)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
                
        # final chunk could be shorter than num_cores, so it's handled waiting for its completion (join without arguments wait for the end of the process)
        if i == len(dwi_files) - 1:
            for p in procs:
                p.join()
        
    logging.info('Parallelization finished.')

        
    