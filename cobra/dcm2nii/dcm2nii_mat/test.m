close all
clear
clc

%%
addpath('./functions/');
dcm_dir='F:\CoBra\Data\dcm\2019_01\00ade8f21e97e455352491aab6b00cb3\7dca7972096ccaa00a348888d2bc22ab';
tgt_dir='F:\CoBra\Data\test\spm';
tgt_file = '00001';
dcm2nii_main(dcm_dir, tgt_dir);
%spm12_main(dcm_dir, tgt_dir);