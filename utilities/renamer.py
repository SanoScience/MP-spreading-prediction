import os
from subprocess import Popen, PIPE, STDOUT

category = input('Type category (i.e. AD, LMCI, EMCI, NC): ')

# a max depth of 3 allows to reach deep folders (i.e. AD_DWI/derivatives/sub-4009) and still not getting files (i.e. sub-4009/ses-baseline/dwi/sub-4009*.nii are 4 levels)
folders = Popen("find . -maxdepth 3 -name \'sub*\'", shell=True, stdout=PIPE)
folders = str(folders.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

# first, rename all folders
for f in folders:
    new_f = f.replace('sub-', f'sub-{category}')
    command = f"mv {f} {new_f}"
    print(command)
    output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)

# '*.*' matches only files with extension (not folders)
search_output = Popen("find . -name \'*.*\'", shell=True, stdout=PIPE)
files = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

for f in files:
    path = f.removesuffix(f.split('/')[-1])
    new_f = f.split('/')[-1].replace('sub-', f'sub-{category}')
    output = ''
    print(f)
    command = f"mv {f} {path+new_f}"
    print(command)
    output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)