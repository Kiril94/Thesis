
close all
clear
clc

%%

addpath('./spm12/');
dcmFiles = dir('D:\Microbleed\Codes\sample\*.dcm');
dcmFiles = strcat({dcmFiles(:).folder}, '\', {dcmFiles(:).name});
headers = spm_dicom_headers(dcmFiles, false);
outFile = spm_dicom_convert(headers, 'all', 'flat', 'nii', 'D:\Microbleed\Codes\', false);
gzip(outFile.files{:});
delete(outFile.files{:});
[filePath, fileName, fileExt] = fileparts(outFile.files{:});
movefile([outFile.files{:}, '.gz'], fullfile(filePath, 'result.nii.gz'));

%%

addpath('./dcm2nii/');
out = dicm2nii('D:\Microbleed\Codes\sample\', 'D:\Microbleed\Codes\', '.nii.gz');
load('D:\Microbleed\Codes\dcmHeaders.mat');
delete('D:\Microbleed\Codes\dcmHeaders.mat');
movefile(['D:\Microbleed\Codes\', char(fieldnames(h)), '.nii.gz'], ['D:\Microbleed\Codes\', 'result', '.nii.gz']);
