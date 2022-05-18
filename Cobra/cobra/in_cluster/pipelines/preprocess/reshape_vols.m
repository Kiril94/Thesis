function [] = reshape_vols(in_main_path,out_main_path,is_mask)

addpath('/home/neus/Documents/09.UCPH/MasterThesis/mostafa_shared/Codes');
%addpath("/home/lkw918/cobra/pipelines/Codes");

%is_mask = true;
%in_main_path = "/home/lkw918/cobra/data/volumetric_data/original/cph_domain/images";
%out_main_path = "/home/lkw918/cobra/data/volumetric_data/reshaped/cph_domain/images";


%in_main_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/MultiResUNet_cmb/cmb_data/volumetric_data/original/test/masks";
%out_main_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/MultiResUNet_cmb/cmb_data/volumetric_data/reshaped/test/masks";


list_files = dir(in_main_path);
file_names = {list_files(3:5).name};

for i=1:length(file_names)

name = char(file_names(i));
in_path = char(strcat(in_main_path,"/",name));
out_path = char(strcat(out_main_path,"/",name));

% Reading the volume from the data store
imageInfo = niftiRead(in_path, 'double', 'header'); % selected image for processing
imageVolume = imageInfo.img; % RAS image volume

% Resizing the volumes to have an isotropic resolution of 1 (mm)
newDims1 = 2 * round(size(imageInfo.img) .* dim_apply_xform(imageInfo.hdr.dime.pixdim(2 : 4), imageInfo.xform) / 2); % nearest scaled even numbers
if ~(is_mask)
    imageVolume = imresize3(imageVolume, newDims1, 'linear', 'Antialiasing', true);
else
    imageVolume = imresize3(imageVolume,newDims1,'nearest');
end

% Padding to create volumes with a minimum dimension of maxPatchSize
maxPatchSize = 256; % maximum patch size
imageVolume = padarray(imageVolume, max(0, (maxPatchSize - newDims1) / 2), 0, 'both');
newDims2 = size(imageVolume);

% Center cropping to create volumes of the same dimension of maxPatchSize
winSize = centerCropWindow3d(newDims2, min(maxPatchSize, newDims2));
imageVolume = imcrop3(imageVolume, winSize);

if ~(is_mask)
    %image normalization and contrast adjustment
    imageVolume = imageVolume - min(imageVolume(:));
    imageVolume = imageVolume / 2 ^ nextpow2(max(imageVolume(:)) + 1);

    pixelRange = stretchlim(imageVolume, [0.0009, 0.9999]);
    imageVolume = imadjustn(imageVolume, [min(pixelRange(1,:)),max(pixelRange(2,:))], [0, 1], 1);

    imageVolume = (imageVolume - min(imageVolume(:))) / (max(imageVolume(:)) - min(imageVolume(:)));

    disp(min(imageVolume(:)));
    disp(max(imageVolume(:)));

else
    imageVolume = (imageVolume - min(imageVolume(:))) / (max(imageVolume(:)) - min(imageVolume(:)));
    disp(unique(imageVolume));

end


% Updating the nifti information for storing
imageInfoNew = imageInfo;
imageInfoNew.img = im2uint16(imageVolume);
imageInfoNew.hdr.dime.dim(2 : 4) = maxPatchSize;
imageInfoNew.hdr.dime.pixdim(2 : 4) = 1;
niftiWrite(imageInfoNew, out_path, 'uint16');

end