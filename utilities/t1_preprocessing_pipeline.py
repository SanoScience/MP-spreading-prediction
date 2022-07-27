import os
from subprocess import Popen, PIPE, STDOUT

print("Looking for all '.nii' files in the current path...")

output = Popen("find . -wholename \'*/anat/*.nii\'", shell=True, stdout=PIPE)
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
    command = f"./preprocessing.sh -i {path+name} -p {path} -f 0.5 -t t1"
    print(f"command: {command}")
    # loading processes queue
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