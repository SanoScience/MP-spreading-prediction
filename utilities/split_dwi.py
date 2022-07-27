import os
from posixpath import join
import re
from subprocess import Popen, PIPE, STDOUT
from collections import defaultdict


#date_re = re.compile(r"\d\d\d\d-\d\d-\d\d")
#date = 'date-'+ date_re.search(name).group()
type = input("Please insert the patient category (i.e. AD, LMCI, EMCI, CN, ...): ")
sub_re = re.compile(rf"sub-{type}\d\d\d\d")
date_re = re.compile(r"\d\d\d\d-\d\d-\d\d")
followup_re = re.compile(r"ses-followup")
dwi_re = re.compile(r"_dwi")

# .nii, .json, .bval, .bvec
data_type = 4

# the following command matches only folders named 'anat' (not files)
search_output = Popen("find . -wholename \'*dwi\'", shell=True, stdout=PIPE)
dwi_dirs = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

dwis = defaultdict(list)
for d in dwi_dirs:
    print(d)
    # find dwi files (.nii and .json) in the considered dwi directory
    sub_dwi = Popen(f"find {d} -name \'*dwi*\'", shell=True, stdout=PIPE)
    sub_dwi = str(sub_dwi.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[1:]
    
    dwi_dict = defaultdict(list) 
    for p in sub_dwi:
        print(p)
        dwi = dwi_re.search(p).group().removesuffix('_dwi')
        # only baseline dwis are moved
        if not followup_re.match(p):
            dwi_dict[dwi].append(p)
    
    for t in dwi_dict.keys():
        # sorting the list by alphabetical order puts the oldest date in first position (ascending order)
        dwi_dict[t].sort()
        
        # if there is more than a single dwi (2 because the regex matches .json and .nii) for that patient...
        while len(dwi_dict[t])>data_type:
            # ... then take the newest [-1] and move it to the followup folder, and iterate until only one remains in the baseline
            sub_id = sub_re.search(d).group()
            followup_dir = sub_id + os.sep + 'ses-followup' + os.sep + 'dwi' + os.sep
            name = dwi_dict[t]
            os.system(f"mkdir -p {followup_dir}")
            
            for _ in range(data_type):
                # moving the last dwi (both .json and .nii) 
                last_dwi = dwi_dict[t][-1]
                new_name = str(last_dwi).split(os.sep)[-1].replace('ses-baseline', 'ses-followup')
                os.system(f"mv {last_dwi} {followup_dir + new_name}")
                print(f"mv {last_dwi} {followup_dir}")
                dwi_dict[t].pop(-1)      
            