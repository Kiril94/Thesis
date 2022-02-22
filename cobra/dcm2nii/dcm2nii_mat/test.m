close all
clear
clc

%%

addpath('./functions/');
dcmdir='F:\CoBra\Data\dcm\2019_01\00ade8f21e97e455352491aab6b00cb3';
tgt_dir='F:\CoBra\Data\test';
tgt_file = '00001';
dcm2nii_main(dcmdir, tgt_dir, tgt_file);
