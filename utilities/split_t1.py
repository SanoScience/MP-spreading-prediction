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
anat_re = re.compile(r"_anat")

data_type = 2

# the following command matches only folders named 'anat' (not files)
search_output = Popen("find . -wholename \'*anat\'", shell=True, stdout=PIPE)
anat_dirs = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

anats_dict = defaultdict(list)
for d in anat_dirs:
    print(d)
    # find pet files (.nii and .json) in the considered anat directory
    sub_anats_dict = Popen(f"find {d} -name \'*anat*\'", shell=True, stdout=PIPE)
    sub_anats_dict = str(sub_anats_dict.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[1:]
    
    images_dict = defaultdict(list) 
    for p in sub_anats_dict:
        print(p)
        anat = anat_re.search(p).group().removesuffix('_anat')
        # only baseline anats_dict are moved
        if not followup_re.match(p):
            images_dict[anat].append(p)
    
    for t in images_dict.keys():
        # sorting the list by alphabetical order puts the oldest date in first position (ascending order)
        images_dict[t].sort()
        
        # if there is more than a single pet (2 because the regex matches .json and .nii) obtained with that tracer (for that patient)...
        # put 3 instead of 3 to deal with derivatives (.json and 2 .nii)
        while len(images_dict[t])>data_type:
            # ... then take the newest [-1] and move it to the followup folder, and iterate until only one remains in the baseline
            sub_id = sub_re.search(d).group()
            followup_dir = sub_id + os.sep + 'ses-followup' + os.sep + 'anat' + os.sep
            name = images_dict[t]
            os.system(f"mkdir -p {followup_dir}")
            
            for _ in range(data_type):
                # moving the last pet (both .json and .nii) 
                last_anat = images_dict[t][-1]
                new_name = str(last_anat).split(os.sep)[-1].replace('ses-baseline', 'ses-followup')
                os.system(f"mv {last_anat} {followup_dir + new_name}")
                print(f"mv {last_anat} {followup_dir}")
                images_dict[t].pop(-1)      
            