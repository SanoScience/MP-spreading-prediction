import os
import multiprocessing
import logging
from tqdm import tqdm
from glob import glob

def task_completion_info(sound_duration=1, sound_freq=440):
    logging.info('Process completed.')
    os.system(f'play -nq -t alsa synth {sound_duration} sine {sound_freq}')  
    
def parallelize(dwi_files, tract_files, num_cores, run, config, general_dir):
    ''' Run function on multiple subjects in parallel. '''

    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    logging.info(f'Parallelization started ({num_cores} cores).')
    
    procs = []
    #sub_re = re.compile("sub-.+/")
    #ses_re = re.compile(r"ses-(baseline|followup)")
    
    #run(stem_dwi = '', stem_anat = '', tractogram = '', config = None, general_dir = '')
    
    if len(tract_files) > 0:
        print("Computing connectivity matrices of already existing tractograms")
        logging.info("Computing connectivity matrices of already existing tractograms")
        for i in range(len(tract_files)):
            logging.info(f"Queueing {tract_files[i]}")
            proc = multiprocessing.Process(target=run, args=('', '', tract_files[i], config, general_dir))
            procs.append(proc)
            proc.start() 
            
            while len(procs)%num_cores == 0 and len(procs) > 0:
                for p in procs:
                    p.join(timeout=10)
                    # when a process is done, remove it from processes queue
                    if not p.is_alive():
                        procs.remove(p)
    
    
    logging.info("Computing tractograms")
    print("Computing tractograms")
    for i in range(len(dwi_files)):
        logging.info(f"Queueing {dwi_files[i]}")
        try:
            stem_t1 = glob(dwi_files[i].split('/dwi/')[0] + os.sep + 'anat' + os.sep + '*_anat.nii.gz')[0].split('.')[0]
        except Exception as e:
            logging.error(f"Error with T1 searching for {dwi_files[i]}. Traceback:")
            logging.error(e)

        proc = multiprocessing.Process(target=run, args=(dwi_files[i].split('.')[0], stem_t1, '', config, general_dir))
        procs.append(proc)
        proc.start()

        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                p.join(timeout=10)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
                
        # final chunk could be shorter than num_cores, so it's handled waiting for its completion (join without arguments wait for the end of the process)
        if i == len(dwi_files) - 1:
            for p in procs:
                p.join()
        
    logging.info('Parallelization finished.')
    return

        
def parallelize_CM(trk_files, num_cores, func, config, general_dir):
    ''' Run function on multiple subjects in parallel. '''

    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    logging.info(f'Parallelization started ({num_cores} cores).')
    
    procs = []
    # instantiating process with arguments
    for i in tqdm(range(len(trk_files))):
        output_dir = trk_files[i].removesuffix(trk_files[i].split(os.sep)[-1])

        proc = multiprocessing.Process(target=func, args=(trk_files[i], config, general_dir, output_dir))
        procs.append(proc)
        proc.start()

        while len(procs)%num_cores == 0 and len(procs) > 0:
            for p in procs:
                p.join(timeout=10)
                # when a process is done, remove it from processes queue
                if not p.is_alive():
                    procs.remove(p)
                
        # final chunk could be shorter than num_cores, so it's handled waiting for its completion (join without arguments wait for the end of the process)
        if i == len(trk_files) - 1:
            for p in procs:
                p.join()
        
    logging.info('Parallelization finished.')