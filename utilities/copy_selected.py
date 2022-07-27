import os
from subprocess import Popen, PIPE, STDOUT

search_output = Popen("find . -name \'*rough.csv\'", shell=True, stdout=PIPE)
files = str(search_output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
output_dir = 'just-CM/'

for f in files:
    name = str(f.split(os.sep)[-1])
    path = str(f.removesuffix(name).removeprefix('./'))
    command = f"mkdir -p {output_dir+path} && cp {f} {output_dir+path+name}"
    print(command)
    output = str(Popen(command, shell=True, stdout=PIPE).stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    print(output)