# !/bin/bash
# preprocess DWI files before generating tractogram from them
# write analogical script to effective_connectivity_toolbox/preprocessing/pre_process.sh

data_dir='/home/bam/Misfolded-protein-spreading/data/input/sharepoint/ADNI/*.nii.gz'
atlas_path='/home/bam/Misfolded-protein-spreading/data/input/atlas_reg.nii.gz'

for filepath in $data_dir; do 
    echo "processing file: " "$filepath"

    filename="$(basename ${filepath} .nii.gz)"

    echo "Eddy current correction"
    eddy_correct $filepath ${filename}_eddycorrected.nii.gz -interp trilinear

    echo "Deobliquing TODO"
    echo "Reorienting TODO"
    echo "Skull stripping TODO"

    echo "Generate a registration map using the refence volume of the atlas"
    flirt -ref ${filename}_eddycorrected.nii.gz -in ${atlas_path} -omat regist_mat.mat 

    echo "Register the atlas using the generated matrix"
    flirt -ref ${filename}_eddycorrected.nii.gz -in ${atlas_path} -applyxfm -init regist_mat.mat -out atlas_registered.nii.gz -interp nearestneighbour
done