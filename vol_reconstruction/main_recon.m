cd(fileparts(mfilename('fullpath')))
addpath(genpath('.'))
gpuDevice(1);

%% Load aberrated data and set universal params
data_name = 'test';
data = XxReadTiffSmallerThan4GB(['../demo_data/test_data/input/' data_name '.tif']);

maxIter = 10; % maximum recon iteration
save_iter = 5; % save interval (in iterations)

%% LEAO recon.
% 1.Compute PSF with estimated aberration
LEAO_model_path = '../aberration_estimation/demo_model/TestResult/';
epoch_detail = 'epoch-best';

aber_path = [LEAO_model_path data_name '_pred_phase_' epoch_detail '.tif'];
phase = XxReadTiffSmallerThan4GB(aber_path);
psf = computePSF(phase,'./PSF/',[data_name '_LEAO']);

% 2.Reconstruction with PSF
DAO_flag = 0;
recon_specify_r('./Recon/',[data_name '_LEAO'],psf,data,DAO_flag,maxIter,save_iter);

%% DAO recon.
% 1.Compute PSF with with 0 aberration
phase = 0;
psf = computePSF(phase,'./PSF/','NoAber');

% 2.Reconstruction with PSF
DAO_flag = 1;
recon_specify_r('./Recon/',[data_name '_DAO'],psf,data,DAO_flag,maxIter,save_iter);

%% w/o AO recon.
DAO_flag = 0;
recon_specify_r('./Recon/',[data_name '_DAO'],psf,data,DAO_flag,maxIter,save_iter);

%% GT Recon.
% 1.Compute PSF with GT aberration
aber_path = [LEAO_model_path data_name '_gt_phase.tif'];
phase = XxReadTiffSmallerThan4GB(aber_path);
psf = computePSF(phase,'./PSF/',[data_name '_GT']);

% 2.Reconstruction with PSF
DAO_flag = 0;
recon_specify_r('./Recon/',[data_name '_GT'],psf,data,DAO_flag,maxIter,save_iter);