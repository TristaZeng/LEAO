function psf=computePSF(phase,save_psf_path,save_psf_name)

mkdir(save_psf_path)

Nnum=       13;
M =         20;% magnification || formula(13)
NA =        0.5;
MLPitch =   100e-6;% microlens pitch size
fml =       2100e-6;% microlens focal length
lambda =    525*1e-9;% laser light wave length
OSR =       3;% calculate times in one pixel
n =         1.0;% refractive index
zspacing = 3.4147*1e-6;
zmax = zspacing*15;
r = 6;
zmin = -zmax;
eqtol = 1e-10;% threshold
tol = 0.0005; % threshold
dxy = MLPitch/Nnum/M/OSR;
k0 = 2*pi*1/lambda; % k

if mod(Nnum,2)==0,
    error(['Nnum should be an odd number']);
end
pixelPitch = MLPitch/Nnum; %% pitch of virtual pixels

x1objspace = [0];
x2objspace = [0];
x3objspace = [zmin:zspacing:zmax]+1e-9; % offset is necessary, otherwise the central slice will be wierd
objspace = ones(length(x1objspace),length(x2objspace),length(x3objspace));% discrete object space

IMGsize = Nnum*41;
IMGsize_L = Nnum*41;
HALF_ML_NUM = 20;

validpts = find(objspace>eqtol);% find non-zero points
numpts = length(validpts);%
[p1indALL p2indALL p3indALL] = ind2sub( size(objspace), validpts);% index to subcripts
p1ALL = x1objspace(p1indALL)';% effective obj points x location
p2ALL = x2objspace(p2indALL)';% effective obj points y location
p3ALL = x3objspace(p3indALL)';% effective obj points z location


disp(['Start Calculating PSF...']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Compute Light Field PSFs (light field) %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pixelPitch_OSR = MLPitch/OSR/Nnum; %simulated pixel size after OSR
fov=length(-(HALF_ML_NUM+1)*OSR*Nnum:(HALF_ML_NUM+1)*OSR*Nnum)*pixelPitch_OSR;   %the size of field of view for the PSF
pixelSize_OSR=length(-(HALF_ML_NUM+1)*OSR*Nnum:(HALF_ML_NUM+1)*OSR*Nnum); %the number of the pixels for each PSF
k2=2*pi/lambda;

fx_sinalpha = 1/(2*pixelPitch_OSR);
fx_step = 1/fov ;
fx_max = fx_sinalpha ;
fx= -fx_max+fx_step/2 : fx_step : fx_max;
[fxcoor fycoor] = meshgrid( fx , fx );
fx2coor=fxcoor.*fxcoor;
fy2coor=fycoor.*fycoor;

aperture_mask=((fx2coor+fy2coor)<=((NA/(lambda*M)).^2));
psfWAVE2=ones(pixelSize_OSR,pixelSize_OSR).*aperture_mask;
psfWAVE2 = gpuArray(single(psfWAVE2));


x1MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2]; % total x space per ML
x2MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2]; % total y space per ML
x1MLdist = length(x1MLspace);%
x2MLdist = length(x2MLspace);%

x1space = (pixelPitch/OSR)*[-(HALF_ML_NUM+1)*Nnum*OSR:1:(HALF_ML_NUM+1)*Nnum*OSR]; % x space
x2space = (pixelPitch/OSR)*[-(HALF_ML_NUM+1)*Nnum*OSR:1:(HALF_ML_NUM+1)*Nnum*OSR]; % y space
x1length = length(x1space);%x
x2length = length(x2space);%y

MLARRAY = calcML(fml, k0, x1MLspace, x2MLspace, x1space, x2space); % micro array phase mask
MLARRAY = gpuArray(single(MLARRAY));

x1objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];% corresponding object space x1
x2objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];% corresponding object space x2
XREF = ceil(length(x1objspace)/2);
YREF = ceil(length(x1objspace)/2);
centerPT = ceil(length(x1space)/2);
halfWidth = HALF_ML_NUM*Nnum*OSR;
CP = ( (centerPT-1)/OSR+1 - halfWidth/OSR :1: (centerPT-1)/OSR+1 + halfWidth/OSR  );%


Hz = zeros(length(CP),length(CP),Nnum,Nnum);
psf = zeros(IMGsize,IMGsize,Nnum,Nnum,numpts,'single');
for eachpt=1:numpts
    aa = tic;
    if(eachpt<0)
        continue;
    else
        disp(['calcu point #',num2str(eachpt),' ...............']);
        %                 time_s = tic;
        p1 = p1ALL(eachpt); % object point #eachpt x
        p2 = p2ALL(eachpt);
        p3 = p3ALL(eachpt);
        
        %                 timeWAVE = tic;
        tempP=k2*n*p3*realsqrt((1-(fxcoor.*lambda./n.*M).^2-(fycoor.*lambda./n.*M).^2).*aperture_mask);
        tempP = gpuArray(single(tempP));
        psfWAVE_fAFTERNO=psfWAVE2.*exp(1j*tempP);
        psfWAVE_AFTERNO=fftshift(ifft2(ifftshift(squeeze(psfWAVE_fAFTERNO))));

        cutoff_freq = 2*NA/lambda;
        dkxy = 1/(dxy*size(psfWAVE_AFTERNO,1));
        aperture_diam_tmp = cutoff_freq/dkxy; 
        aperture_diam_tmp = aperture_diam_tmp*(2*r+1)/Nnum;
        aperture_diam = floor(aperture_diam_tmp/2)*2+1;
        aber_phase = imresize(phase,[aperture_diam,aperture_diam]);
        aber_phase(~isfinite(aber_phase)) = 0;
        aber_phase(isnan(aber_phase)) = 0;
        aber_phase2=padarray( aber_phase,[(size(psfWAVE_AFTERNO,1)-size(aber_phase,1))/2,(size(psfWAVE_AFTERNO,2)-size(aber_phase,2))/2] );
        psfWAVE_AFTERNO = ifft2(ifftshift(fftshift(fft2(psfWAVE_AFTERNO)).*exp(1j.*aber_phase2)));
        

        for b1 = 1:length(x2objspace)
            for a1 = 1:length(x1objspace)
                psfSHIFT0= im_shift2_GPU(psfWAVE_AFTERNO, OSR*(a1-XREF), OSR*(b1-YREF) );%
                f1=fresnel2D_GPU(psfSHIFT0.*MLARRAY, pixelPitch/OSR, fml,lambda);%
                f1= im_shift2_GPU(f1, -OSR*(a1-XREF), -OSR*(b1-YREF) );%
                [f1_AP_resize, x1shift, x2shift] = pixelBinning_GPU(abs(f1).^2, OSR);
                f1_CP = f1_AP_resize( CP - x1shift, CP-x2shift );
                Hz(:,:,a1,b1) = gather(f1_CP);%
            end
        end
        
        H4Dslice = Hz;
        H4Dslice(find(H4Dslice< (tol*max(H4Dslice(:))) )) = 0;% remove noise
        Hz = H4Dslice;
        
        
        for b2 = 1:length(x2objspace)
            for a2 = 1:length(x1objspace)
                sss = Hz(:,:,a2,b2);
                Hz(:,:,a2,b2) = Hz(:,:,a2,b2)./sum(sss(:));
            end
        end
        
        
            
        filepath = [save_psf_path,save_psf_name,'_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_dz',num2str(zspacing*1e+6),'u'];
        H_z = single(Hz);
        
        disp('split');
        
        border=fix(IMGsize_L/2)-fix(size(H_z,1)/2);
        blur_image=zeros(IMGsize_L,IMGsize_L,size(H_z,3),size(H_z,4));
        
        for i=1:size(H_z,3)
            for j=1:size(H_z,4)
                temp=zeros(IMGsize_L,IMGsize_L);
                temp(border+1:end-border,border+1:end-border)=squeeze(H_z(:,:,i,j));
                blur_image(:,:,i,j)=(im_shift3d(temp,i-((Nnum+1)/2),j-((Nnum+1)/2)));
            end
        end
        
        blur_image(isnan(blur_image)) = 0;
        maxH_z  = max(blur_image(:));
        Output=uint16(blur_image./maxH_z*65535);
        bbb = realign(Output,Nnum,Nnum);
        
        tmp = bbb((IMGsize_L+1)/2-(IMGsize-1)/2:(IMGsize_L+1)/2+(IMGsize-1)/2,(IMGsize_L+1)/2-(IMGsize-1)/2:(IMGsize_L+1)/2+(IMGsize-1)/2,:,:);
        psf_z = single(tmp);

        onetime = toc(aa);
        disp(['idz = ',num2str(eachpt),', taketime ',num2str(onetime),' sec......']);

        psf(:,:,:,:,eachpt) = psf_z;
            
        
    end
    
end
clear Output bbb tmp blur_image H_z Hz
save([filepath,'.mat'],'psf','-v7.3');

