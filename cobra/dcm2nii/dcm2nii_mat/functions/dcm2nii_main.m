function out = dcm2nii_main(src_dir, tgt_dir)
%DCM2NII Converts dcm files to nii.gz using dcm2nii
%   src: directory, where the dicom files of a series are located 
%   tgt: target dir of the dicom series
%   tgt_name: name of the nii file (6 digits), the output will be:
%               tgt_name.nii.gz
addpath('../dcm2nii/');
out = dicm2nii(src_dir, tgt_dir, '.nii.gz');
end

