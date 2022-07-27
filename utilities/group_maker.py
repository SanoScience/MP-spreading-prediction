import os
from subprocess import Popen, PIPE, STDOUT
import json
import re
from collections import defaultdict

print("Looking for all 'dwi.nii' files in the derivatives subfolders...")

output_dwi = Popen("find . -wholename \'*derivatives*/*dwi.nii\'", shell=True, stdout=PIPE)
files_dwi = str(output_dwi.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

output_t1 = Popen("find . -wholename \'*derivatives*/*t1_mask.nii\'", shell=True, stdout=PIPE)
files_t1 = str(output_t1.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

print(f"dwis {len(files_dwi)}")
print(f"t1s {len(files_t1)}")

groups = defaultdict(list)

for dwi in files_dwi:
    dwi = dwi.removeprefix("\"").removeprefix("\'").removesuffix("\"")
    print(f"file: {dwi}")
    
    path = dwi.removesuffix(dwi.split('/')[-1])
    name = (dwi.split('/')[-1]).split('.')[0] # just the patient name (without extension)
    sub_id = name.split('_')[0]
    
    json_data = json.load(open(path+name+'.json'))
    
    #group_name = str(json_data["ManufacturersModelName"]).replace(' ', '_')
    
    volumes = Popen(f"fslhd {dwi} | grep dim4", shell=True, stdout=PIPE)
    vol = str(volumes.stdout.read()).removeprefix('b\'').replace('\'', '').removesuffix('\\n').replace('\\t', ' ').replace('\\n', '  ').split('  ')
    group_name = vol[1]

    t1 = ''
    for f_t in files_t1:
        if re.match(rf".*{sub_id}.*", f_t):
            t1 = f_t
            break
    
    groups[group_name].append(f"{dwi},{t1}")

os.system("mkdir -p groups")

for m in groups.keys():
    group_file = open(f"groups/{m}({len(groups[m])}).csv", "w")
    for line in groups[m]:
        group_file.write(f"{line}\n")
    group_file.close()

print("DONE")