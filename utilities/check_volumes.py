import os
from subprocess import Popen, PIPE, STDOUT

output = Popen("find . -name \'*.bval\'", shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

for f in files:
    command = f"wc {f}"
    output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)