# dMRI_Toolkit
The present repository contains a collection of scripts used to handle and process ADNI MRIs.
----------------
Please use python3.9 to run *_preprocessing_pipeline scripts.
- "aal.nii.gz" is an atlas used in preprocessing phase to register MRIs
- "bids_structure_creator.py" is used to convert an input folder containing DICOM images in a BIDS folder. BIDS format differentiate data in different subfolders (i.e. "dwi", "t1" and "pet")
- "preprocessing.sh" apply preprocessing steps to a single input image (.nii). Run without options to get its API. Using its flags, its behavior can be changed to perform different actions on input images.
- "dwi_preprocessing_pipeline.py" calls repeteadly 'preprocessing.sh' on all .nii files it founds in the 'dwi' subdirectories of its path, using specific parameters to optimize .nii files.
- "t1_preprocessing_pipeline.py" calls repeteadly 'preprocessing.sh' on all .nii files it founds in the 't1' subdirectories, using specific parameters to optimize .nii files. (it includes FAST command, generating PVE masks)
- "check_volumes.py" scan sub-directories to get 'bval' files, showing the number of volumes of each image.
- "count_niis.py" is an utility script for iteratively checking the presence of a particular kind of image (typically, PET ones) in respective patients' folders.
