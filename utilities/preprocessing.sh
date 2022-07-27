#!/bin/bash

# simple bash script to accept a '.nii'/'.nii.gz' file (-i flag) and preprocessing it. output and reference atlas can be accessed with relative flags (-o and -a, respectively). The input file is subjected to bet, fslmaths and eddy_correct preprocessing, before being registered through flirt. The intermediary files produced by these passages can be kept specifying the -k option, otherwise they are cleaned up at the end of the script.

input=''
output=''
output_path = ''
atlas='aal.nii.gz'
keep='n'
path=''
type='dwi'
force=0.4
lpca='n'

while getopts i:a:k:f:p:t:l flag
do
    case "${flag}" in
        i) 
            input=${OPTARG}
            output="derivatives/${input}.nii.gz"
        ;;
        a)
            atlas=${OPTARG}
        ;;
        k)
            keep=${OPTARG}
        ;;
        f)
            force=${OPTARG}
        ;;
        p)
            path=${OPTARG}
        ;;
        t)
            type=${OPTARG}
        ;;
        l)
            lpca=${OPTARG}
        ;;
    esac
done

echo "input: $input"
echo "output: $output"

if [ -z "$input" ] 
then
    echo "Expected argument(s). Options: 
    -i to specify input image WITHOUT EXTENSION (mandatory)
    -a to specify atlas to use for registration ('aal.nii.gz' by default)
    -k to keep intermediary outputs [y/n] (for debugging purposes. By default it's disabled 'n')
    -f to specify the force used for BET (by default 0.4)
    -p to specify the path to the temporary files (don't use it to generate temporary files in the current folder). Please use a subfolder of the current path, without the './' in the beginning (i.e. avoid using './derivatives/')
    -t to declare preprocessing type [dwi/t1/pet] (dwi by default)
    -l to apply lpca denoising [y] or not [n, default] (just for dwi)
    ";
    exit 0;
fi

input_image="${input}.nii"
input_bvec="${input}.bvec"
input_bval="${input}.bval"
input_json="${input}.json"

acqparams="derivatives/${path}acqparams.txt"
index="derivatives/${path}index.txt"

# input_image -> denoise_output -> degibbs_output -(+ BET mask)> fslmaths_output -(+BET mask/acqparams/index/bvecs/bvals)> eddy_output -(+ bvecs/bvals)>  dbc_output -(+ atlas)> output
denoise_output="derivatives/${path}dwi_lpca.nii.gz"
degibbs_output="derivatives/${path}dwi_unrig.nii.gz"
bet_output="derivatives/${input}_bet_out"
bet_output_mask="derivatives/${input}_bet_out_mask"
fslmaths_output="derivatives/${input}_fslm_out.nii.gz"
mat="derivatives/${input}_transformation_matrix.mat" 
eddy_output="derivatives/${input}_eddy_outlier_free_data.nii.gz"
dbc_output="derivatives/${input}_dbc.nii.gz"

cerebellum_image="derivatives/${input}_cerebellum_image.nii.gz"
cerebellum_mask="derivatives/${input}_cerebellum_mask.nii.gz"

if [ "$type" == "dwi" ]; then
    if [ "$lpca" == "y" ]; then
        echo "dipy denoising (LPCA)"
        dipy_denoise_lpca --out_dir $path $input_image $input_bval $input_bvec
        input_image=$denoise_output
    fi

    echo "degibbs"
    dipy_gibbs_ringing $input_image --num_processes 8 --out_dir "derivatives/${path}"
    input_image=$degibbs_output
fi

echo "bet"
bet2 $input_image $bet_output -g -0.5 -m -f $force

echo "fslmaths"
fslmaths $input_image -mas "${bet_output_mask}.nii.gz" $fslmaths_output

flirt_input="$fslmaths_output"

if [ "$type" == "dwi" ]; then

    echo "eddy_cuda"
    eddy_cuda10.2 --imain=$fslmaths_output --mask=$bet_output_mask --acqp=$acqparams --index=$index --bvecs=$input_bvec --bvals=$input_bval --out=$eddy_output --repol --interp=trilinear
    # --repol "Detect and replace outlier slices (default false)"
    # --json option has been omitted because it's buggy (it crashes if it doesn't find an expected field) 
    # slice-to-volume correction is enabled (default trilinear) for 5 interactions (--s2v_niter default value)
    # eddy_cuda10.2 --imain=fslm_out.nii.gz --mask=bet_out_mask.nii.gz --acqp=acqparams.txt --index=index.txt --bvecs=sub-AD4009_ses-1_acq-AP_dwi.bvec --bvals=sub-AD4009_ses-1_acq-AP_dwi.bval --out=eddy_out.nii.gz    
    # output would be named 'eddy_outlier_free_data.nii.gz'
    #eddy_correct $fslmaths_output $fslmaths_output -interp=trilinear

    #echo "B1 field inhomogeneity correction"
    #dwibiascorrect ants $eddy_output $dbc_output -fslgrad $input_bvec $input_bval -nthreads 8
    #flirt_input ="$dbc_output"
fi

if [ "$type" == "pet" ]; then 
    echo "flattening 4D pet into 3D mean values image"
    fslmaths $flirt_input -Tmean $flirt_input
fi

echo "creating matrix"
flirt -ref $atlas -in $flirt_input -cost mutualinfo -searchcost mutualinfo -omat $mat

echo "atlas registration"
flirt -ref $atlas -in $flirt_input -applyxfm -init $mat -out $output 

if [ "$type" == "pet" ]; then 

    echo "cerebellum extraction"
    fslmaths $atlas -uthr 90 $cerebellum_mask
    fslmaths $output -mas $cerebellum_mask $cerebellum_image

    echo "subtracting mean of cerebellum to the rest"
    fslmaths $output -sub `fslstats $cerebellum_image -M` $output

    echo "removing negative values"
    fslmaths $output -thr 0 $output
fi

if [ "$type" == "t1" ]; then 
    echo "fast"
    fast -o $output $output
fi

if [ "$keep" == "n" ]; then
    echo "Cleaning (please ignore possible errors)"
    rm $denoise_output
    rm $degibbs_output
    rm ${path}*bet_out*
    rm $fslmaths_output
    rm $eddy_output
    rm -r dwibiascorrect-tmp*
    rm derivatives/${path}*eddy*
    rm $mat
    rm $dbc_output
    rm $cerebellum_image
    rm $cerebellum_mask
    echo "Cleaned";
fi

echo "Done. $output completed"