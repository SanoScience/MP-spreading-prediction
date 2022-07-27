import os
from subprocess import Popen, PIPE, STDOUT

output = Popen("find . -maxdepth 1 -wholename \'*sub*\'", shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
print(files)
for f in files:
    command = f"find {f} -name \'*_pet*.nii\' | wc -l"
    output = Popen(command, shell=True, stdout=PIPE)
    pets = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n')
    print(f + ': ' + str(pets))