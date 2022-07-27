import os
from subprocess import Popen, PIPE, STDOUT
import json

print("Looking for all '.nii' files in the current path...")

output = Popen("find . -wholename \'*/pet*/*.nii\'", shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

print("Found {} files".format(len(files)))
print(files)

print("Starting preprocessing...")
processes = []
done = 0
for f in files:
    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    print(f"file: {f}")
    path = f.removesuffix(f.split('/')[-1])
    name = (f.split('/')[-1]).split('.')[0] # just the patient name (without extension)

    print("path: ", path)
    # 'name' will not be overwritten because output of preprocessing is .nii.gz (input image is expected to be a .nii)
    command = f"./preprocessing.sh -i {path+name} -p {path} -t pet -f 0.4"
    # loading processes queue
    print(f"command: {command}")
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, executable="/bin/bash")
    processes.append(p)
    done = done + 1
    # every 8 files stop and wait
    if len(processes)%8==0:
        for p in processes:
            p.wait()
        print(f"{done}/{len(files)}")
        
	
for p in processes:
    p.wait()

print("PREPROCESSING DONE")