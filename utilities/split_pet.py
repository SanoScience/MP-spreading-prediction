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
trc_re = re.compile(r"_trc-.*_pet")

data_type = int(input('Insert \'2\' if this is original data, \'3\' if it is preprocessed'))

# the following command matches only folders named 'pet' (not files)
search_output = Popen("find . -wholename \'*pet\'", shell=True, stdout=PIPE)
pet_dirs = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

pets = defaultdict(list)
for d in pet_dirs:
    print(d)
    # find pet files (.nii and .json) in the considered pet directory
    sub_pets = Popen(f"find {d} -name \'*pet*\'", shell=True, stdout=PIPE)
    sub_pets = str(sub_pets.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[1:]
    
    tracers_dict = defaultdict(list) 
    for p in sub_pets:
        print(p)
        trc = trc_re.search(p).group().removeprefix('_trc-').removesuffix('_pet')
        # only baseline pets are moved
        if not followup_re.match(p):
            tracers_dict[trc].append(p)
    
    for t in tracers_dict.keys():
        # sorting the list by alphabetical order puts the oldest date in first position (ascending order)
        tracers_dict[t].sort()
        
        # if there is more than a single pet (2 because the regex matches .json and .nii) obtained with that tracer (for that patient)...
        # put 3 instead of 3 to deal with derivatives (.json and 2 .nii)
        while len(tracers_dict[t])>data_type:
            # ... then take the newest [-1] and move it to the followup folder, and iterate until only one remains in the baseline
            sub_id = sub_re.search(d).group()
            followup_dir = sub_id + os.sep + 'ses-followup' + os.sep + 'pet' + os.sep
            name = tracers_dict[t]
            os.system(f"mkdir -p {followup_dir}")
            
            for _ in range(data_type):
                # moving the last pet (both .json and .nii) 
                last_pet = tracers_dict[t][-1]
                new_name = str(last_pet).split(os.sep)[-1].replace('ses-baseline', 'ses-followup')
                os.system(f"mv {last_pet} {followup_dir + new_name}")
                print(f"mv {last_pet} {followup_dir}")
                tracers_dict[t].pop(-1)      
            