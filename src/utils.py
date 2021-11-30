import os
from multiprocessing import Process
import logging


def task_completion_info(sound_duration=1, sound_freq=440):
    print('Process completed.')
    os.system(f'play -nq -t alsa synth {sound_duration} sine {sound_freq}')
    
    
def parallelize(subject_names, func, args):
    ''' Run specific function on multiple subjects in parallel. '''
    
    logging.info('Parallelization started.')
    
    procs = []
    proc = Process(target=func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in subject_names:
        proc = Process(target=func, args=(args, name))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
        
    logging.info('Parallelization finished.')

        
    