function recon_specify_r(savedir,save_prefix,psf,data,AOstar,maxIter,save_iter)

%% paramaters
% optical params (determined when generating simulation data)
Nshift=6;
Nnum=13;
thresh=1e-10;
r=6;
% recon params
min_prc=0.1;
max_prc=99.999;
half_Nnum=ceil(Nnum/2);
max_weight_for_DAO=r^2;
max_weight=r^2; %% angle range used for reconstruction
Nbx=1;%%% =1 means the aberration of the whole field of view is consistent
Nby=1;%%% =1 means the aberration of the whole field of view is consistent
rm_defocus = 1;

mkdir(savedir);
savepath=strcat(savedir,save_prefix,'_r',num2str(r),'Nshift',num2str(Nshift));

weight1=squeeze(sum(sum(sum(psf,1),2),5)); 
[xx,yy]=meshgrid(-floor(Nnum/2):floor(Nnum/2),-floor(Nnum/2):floor(Nnum/2));
round1=zeros(Nnum,Nnum);
round1(xx.^2+yy.^2<=max_weight)=1;
weight1=weight1.*round1;
weight1=weight1.*0.8./max(weight1(:));

%% data manipulation
cnt=0;
if size(data,3)==Nnum*Nnum
    for u=1:Nnum
        for v=1:Nnum
            cnt=cnt+1;
            WDF(:,:,u,v) = data(:,:,cnt);
        end
    end
else
    if size(data,3)==49
        r_wdf = 4;
    elseif size(data,3)==81
        r_wdf = 5;
    elseif size(data,3)==113
        r_wdf = 6;
    elseif size(data,3)==317
        r_wdf = 10;
    end
    for u=1:Nnum
        for v=1:Nnum
            if ((u-half_Nnum)^2+(v-half_Nnum)^2)<=r_wdf^2 
                cnt=cnt+1;
            end
            if ((u-half_Nnum)^2+(v-half_Nnum)^2)<=max_weight
                WDF(:,:,u,v) = data(:,:,cnt);
            else
                WDF(:,:,u,v) = zeros(size(data,1),size(data,2));
            end
        end
    end 
end
WDF=imresize(WDF,[size(WDF,1)*Nnum/Nshift,size(WDF,2)*Nnum/Nshift]);

        
%% phase space reconstruction
Xguess=ones(size(WDF,1),size(WDF,2),size(psf,5));
Xguess=Xguess./sum(Xguess(:)).*sum(WDF(:))./(size(WDF,3)*size(WDF,4));

deconvRLGPU_20200306_phasespace(maxIter,Xguess,WDF,psf,savepath,...
AOstar,thresh,weight1,max_weight_for_DAO,rm_defocus,Nbx,Nby,save_iter,min_prc,max_prc);