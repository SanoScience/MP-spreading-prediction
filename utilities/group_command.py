import os
from subprocess import Popen, PIPE, STDOUT

search_output = Popen("find . -maxdepth 3 -wholename \'*ses-followup/*\'", shell=True, stdout=PIPE)
files = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

for f in files:
    #name = f.replace('ses-baseline', 'ses-followup')
    output = ''
    print(f)
    new_f = f.replace('ses-followup/','ses-followup/pet/')
    #command = f"mkdir {new_f.removesuffix(new_f.split('/')[-1])} && mv {f} {new_f}"
    #print(command)
    #output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)