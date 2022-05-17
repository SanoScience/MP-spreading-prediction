#!/bin/sh

# Place this script inside the derivatives folder to clean temporary files

rm -r `find . -name '*intermediate'`
rm `find . -name '*mat'`
rm `find . -name 'first_slice.nii.gz'`
rm `find . -name '*eddy*'`
rm `find . -name 'stat_result.json'`