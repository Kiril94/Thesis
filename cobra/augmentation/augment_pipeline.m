%restoredefaultpath
close all
clear
clc

%% Original Image

D = niftiread("C:\Users\kiril\Thesis\CoBra\cobra\data\augmentations\MICCAI\imagesTr\MICCAI_000_0000.nii.gz");
Dy = niftiread("C:\Users\kiril\Thesis\CoBra\cobra\data\augmentations\MICCAI\labelsTr\MICCAI_000_0000.nii.gz");
X = squeeze(D); % input 3D MRI volume
Y = squeeze(Dy);
figure
subplot(3,3,1)
imshow(X(:,:,100),[]);
title('Original Image');
xmin = double(min(X, [], 'all'));
xmax = double(max(X, [], 'all'));
X1 = (double(X)-xmin)./(xmax-xmin); 

subplot(3,3,2);
imshow(X1(:,:,100),[]);
title('Intensity MinMax scaled');

subplot(3,3,3)
X2 = X1.*(xmax-xmin)+xmin;
imshow(X2(:,:,100), []);
title('Scaled back');


func_list = {@biasField, @elasticDeform, @gibbsRinging, @motionGhosting};
%%
%%Change params and apply the same to y, scale image first.
num_augmentations = randi(4);
display(num_augmentations);
Xt = X1;
Yt = Y;
for i = 1:num_augmentations
    augmentation = randi(4);
    if augmentation==1
        Xt = biasField(Xt);
    elseif augmentation==2
        sigma = (30 - 20) * rand + 20; % elasticity coefficient randomly chosen in [10 20]
        alpha = (500 - 200) * rand + 200; % scaling factor randomly chosen in [100 200]
        interMethod = 'linear'; % interpolation method, linear for image, nearest for labels
        extraMethod = 'nearest'; % extrapolation method
        Xt = elasticDeform(Xt, sigma, alpha, interMethod, extraMethod); % image deformation
        Yt = elasticDeform(Yt, sigma, alpha, interMethod, extraMethod); % image deformation
    elseif augmentation==3
        cutFreq = randi([96, 128]); % cutting k-space frequency
        dim = randi([1,3]); % performed artefact dimension
        Xt = gibbsRinging(Xt, cutFreq, dim); % image truncation
    else 
        alpha = (0.95 - 0.85) * rand + 0.85; % intensity factor randomly chosen in [0.5 0.7]
        numReps = randi([2, 4]); % number of ghosts
        dim = randi([1,3]); % performed artefact dimension
        Xt = motionGhosting(Xt, alpha, numReps, dim); % image repetition
    end
    display(augmentation);
    aug_func = func_list{augmentation};
    Xt = aug_func(Xt);%
end

