#Import modules
import nipype
import nipype.interfaces.afni        as afni
import nipype.interfaces.freesurfer  as fs
import nipype.interfaces.ants        as ants
import nipype.interfaces.fsl         as fsl
import nipype.interfaces.nipy        as nipy
import nipype.interfaces.spm         as spm
from nipype.interfaces.slicer import BRAINSFit, BRAINSResample
import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.fsl import Info
import nipype.interfaces.petpvc

import os  # system functions
from subprocess import Popen, PIPE, STDOUT


#Specify experiment specifc parameters
input_dir = os.path.realpath('.')

# Loading Subjects list
output = Popen("find . -maxdepth 2-wholename \'*derivatives/sub*\'", shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

pet_list = []
for f in files:
    pet_list.append(f.removeprefix('./'))

# Loading T1 masks list
output = Popen("find . -wholename \'*_pveseg.nii.gz\'", shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
masks_list = []
for f in files:
    masks_list.append(f.removeprefix('./'))

# location of template file
template = Info.standard_image('aal.nii.gz')

#Where can the raw data be found?
grabber = nipype.BIDSDataGrabber()
grabber.inputs.template = '*derivatives*(_pet.nii|_pveseg.nii.gz)*'

#Where should the output data be stored at?
sink = nipype.DataSink()
sink.inputs.base_directory = input_dir + '/derivatives'


#Create a node for each step of the analysis

#Motion Correction (AFNI)
realign = afni.Retroicor()

#Coregistration (FreeSurfer)
coreg = fs.BBRegister()

#Normalization (ANTS)
normalize = ants.WarpTimeSeriesImageMultiTransform()

#Smoothing (FSL)
smooth = fsl.SUSAN()
smooth.inputs.fwhm = 6.0

#Model Specification (Nipype)
modelspec = nipype.SpecifyModel()
modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = 2.5
modelspec.inputs.high_pass_filter_cutoff = 128.0

# Partial Volume Correction
pvc = PETPVC()
pvc.inputs.in_file   = '*_pet.nii'
pvc.inputs.mask_file = '*_pveseg.nii.gz'
pvc.inputs.out_file  = 'pet_pvc_rbv.nii.gz'
pvc.inputs.pvc = 'RBV'
pvc.inputs.fwhm_x = 2.0
pvc.inputs.fwhm_y = 2.0
pvc.inputs.fwhm_z = 2.0
outs = pvc.run() 


#Create a workflow to connect all those nodes
analysisflow = nipype.Workflow()

#Connect the nodes to each other
analysisflow.connect( [ (grabber     ->  realign)    , (realign     ->  coreg      ), (coreg       ->  normalize  ), (normalize   ->  smooth     ), (smooth      ->  modelspec  ), (modelspec   ->  pvc   ), (pvc    ->  sink)       ])

#Run the workflow in parallel
analysisflow.run(mode='parallel')