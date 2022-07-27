import os
import re
import sys

cur_path = os.path.dirname(os.path.realpath(__file__))
if len(sys.argv)<3:
    print("Pass one of the following argument (or even a different one:")
    print("pet-abeta-ac\npet-tau\npet-abeta-av\npet-abeta-fbb\npet-tau\npet-abeta-pib")
    print("and the kind of file you want to enumerate (i.e. nii, nii.gz, ...)")
    quit()

name = fr".*{sys.argv[1]}.*"
for subdir, dirs, files in os.walk(cur_path):
    if files and re.match(name, subdir):
        os.system(f'echo "{subdir}: `find {subdir} -name \'*{sys.argv[2]}\' | wc -l`"')