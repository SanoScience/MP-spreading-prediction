#!/bin/sh

# Place this script inside the derivatives folder to clean temporary files

rm -r `find . -name '*intermediate'`
rm -r `find . -name '*mat'`
rm -r `find . -name '*first_slice.nii.gz'`
rm `find . -name '*eddy*'`
rm `find . -name 'stat_result.json'`

rm `find . -name 'MAR_*.png'`
rm `find . -name 'MAR_*.csv'`
rm `find . -name 'MAR_*.txt'`

rm `find . -name 'ESM_*.png'`
rm `find . -name 'ESM_*.txt'`

rm `find . -name 'NDM_*.png'`
rm `find . -name 'NDM_*.txt'`