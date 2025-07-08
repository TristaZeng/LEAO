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
recon_specify_r('./Recon/',[data_name '_woAO'],psf,data,DAO_flag,maxIter,save_iter);

%% Visualize all recon.
LEAO_recon = XxReadTiffSmallerThan4GB(['./Recon/' data_name '_LEAO_r6Nshift6_iter' num2str(maxIter) '_xcorr_thresh1e-10.tif']);
DAO_recon = XxReadTiffSmallerThan4GB(['./Recon/' data_name '_DAO_r6Nshift6_iter' num2str(maxIter) '_xcorr_thresh1e-10.tif']);
woAO_recon = XxReadTiffSmallerThan4GB(['./Recon/' data_name '_woAO_r6Nshift6_iter' num2str(maxIter) '_xcorr_thresh1e-10.tif']);
GT_vol = XxReadTiffSmallerThan4GB(['../demo_data/test_data/gt_vol/' data_name '_gt_vol.tif']);
cut = 30;
LEAO_MIP = max(LEAO_recon(cut+1:end-cut,cut+1:end-cut,:),[],3);
DAO_MIP = max(DAO_recon(cut+1:end-cut,cut+1:end-cut,:),[],3);
woAO_MIP = max(woAO_recon(cut+1:end-cut,cut+1:end-cut,:),[],3);
GT_MIP = max(GT_vol(cut+1:end-cut,cut+1:end-cut,:),[],3);
fig = figure();
subplot(2,2,1);imshow(XxNorm(LEAO_MIP,0.1,99));title('LEAO');
subplot(2,2,2);imshow(XxNorm(DAO_MIP,0.1,99));title('DAO');
subplot(2,2,3);imshow(XxNorm(woAO_MIP,0.1,99));title('w/o AO');
subplot(2,2,4);imshow(XxNorm(GT_MIP,0.1,99));title('GT');
saveas(fig, './Recon/output.png');
close(fig);