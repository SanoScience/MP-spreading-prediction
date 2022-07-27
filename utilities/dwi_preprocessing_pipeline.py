import os
from subprocess import Popen, PIPE, STDOUT
import json

print("Looking for all '.nii' files in the current path...")

output = Popen("find . -wholename \'*/dwi/*.nii\'", shell=True, stdout=PIPE)
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
    
    json_data = json.load(open(path+name+'.json'))
    image_orientation = json_data["ImageOrientationPatientDICOM"]
    
    # 'EchoSpacing' can be 'Effective' or 'EstimatedEffective'
    es_codename = "EffectiveEchoSpacing"
    if es_codename not in json_data.keys():
        es_codename = "Estimated"+es_codename
        if es_codename not in json_data.keys():
            es_codename = "EchoTime"
    ees = json_data[es_codename]
    amPE = json_data["AcquisitionMatrixPE"]
    freq = round(ees * (amPE - 1), 4)
    acq_f = open(path+"acqparams.txt", "w")
    acq_f.write(f"{image_orientation[0]} {image_orientation[1]} {image_orientation[2]} {freq}\n")
    acq_f.write(f"{image_orientation[3]} {image_orientation[4]} {image_orientation[5]} {freq}\n")
    acq_f.close()
    
    try:
        index = open(path+"index.txt", "w")
        bvals = open(path+name+".bval", "r")
    except Exception as e:
        print("bval not found, skipping...")
        continue
    
    volumes = len(bvals.readline().split(' '))
    for v in range(volumes):
        index.write('1 ' if v%2==0 else '2 ')
    index.close()

    print("path: ", path)
    # 'name' will not be overwritten because output of preprocessing is .nii.gz (input image is expected to be a .nii)
    command = f"./preprocessing.sh -i {path+name} -p {path} -t dwi -l"
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