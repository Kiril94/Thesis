function outFile = spm12_main(src_dir, tgt_dir)
%SPM12_MAIN Summary of this function goes here
%   Detailed explanation goes here
%addpath('../spm12/');
dcmFiles = dir(strcat(src_dir,'\*.dcm'));
dcmFiles = strcat({dcmFiles(:).folder}, '\', {dcmFiles(:).name});
headers = spm_dicom_headers(dcmFiles, false);
outFile = spm_dicom_convert(headers, 'all', 'flat', 'nii', tgt_dir, false);
gzip(outFile.files{:});
delete(outFile.files{:});
[filePath, ~, ~] = fileparts(outFile.files{:});
movefile([outFile.files{:}, '.gz'], fullfile(filePath, 'result.nii.gz'));
end

