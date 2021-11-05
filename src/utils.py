import os

def task_completion_info(sound_duration=1, sound_freq=440):
    print('Process completed.')
    os.system(f'play -nq -t alsa synth {sound_duration} sine {sound_freq}')