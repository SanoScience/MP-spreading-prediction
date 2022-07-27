import os
from subprocess import Popen, PIPE, STDOUT

search_output = Popen("find . -name \'*.json\'", shell=True, stdout=PIPE)
files = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

for f in files:
    #name = f.replace('ses-baseline', 'ses-followup')
    output = ''
    print(f)
    new_f = 'derivatives/' + f.removeprefix('./')
    command = f"cp {f} {new_f}"
    print(command)
    output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)