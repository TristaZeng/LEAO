function Xguess = deconvRLGPU_20200306_phasespace(maxIter,Xguess,blur_image, psf, ...
    savepath,AOstar,thresh,recon_weight,max_angle_for_DAO,rm_defocus,Nbx,Nby,save_iter,min_prc,max_prc)
if nargin<10,rm_defocus=1;end
if nargin<11,Nbx=1;end
if nargin<12,Nby=1;end
if nargin<13,save_iter = 1;end
if ~exist('min_prc','var'),min_prc = 0.1;end
if ~exist('max_prc','var'),max_prc = 99.999;end

Nnum=size(blur_image,3);
[img_r,img_c,~] = size(blur_image);
[psf_r,psf_c,allu,allv,allz] = size(psf);
index_c = [1,allz+1];
% index_c = [1+25,allz+1-25];

blur_image = single(blur_image);
recon_weight = gpuArray(single(recon_weight));
DAO_angles=ones(Nnum,Nnum);
half_Nnum = floor(Nnum/2);
x=[-half_Nnum:half_Nnum];
[xx,yy]=meshgrid(x,x);
DAO_angles(xx.^2+yy.^2>max_angle_for_DAO)=0;
Xguess=gpuArray(single(Xguess));

load(['wjm_map_' num2str(Nnum) 'x' num2str(Nnum) '.mat']);

for i=1:maxIter
    if AOstar>0 && i>=2
        map_wavshape=zeros(Nnum,Nnum,Nbx,Nby,2);
        map_wavshape_weight=ones(Nnum,Nnum,Nbx,Nby);
        sidelobe=100; 
        if mod(img_r,2)
            N3=fix( 1*img_r/(Nbx)/2 )*2+1;
        else
            N3 = fix( 1*img_r/(Nbx)/2 )*2;
        end
        borderx=(img_r-N3*Nbx)/2;
        bordery=(img_c-N3*Nby)/2;
        [coordinate1,coordinate2]=meshgrid(1:img_r,1:img_c);
        x=[-half_Nnum:half_Nnum];
        [Sx,Sy]=meshgrid(x,x);
        for u=1:Nnum
            for v=1:Nnum
                if ((u-ceil(Nnum/2))^2+(v-ceil(Nnum/2))^2)>max_angle_for_DAO
                    Sx(u,v)=0;
                    Sy(u,v)=0;
                end
            end
        end
        
        
        
        for u_2=1:Nnum
            for v_2=1:Nnum
                u=wjm_map_u((u_2-1)*Nnum+v_2);
                v=wjm_map_v((u_2-1)*Nnum+v_2);
                if DAO_angles(u,v)==0
                    continue;
                else
                    sumupXG = gpuArray.zeros(img_r,img_c,'single');
                    for block = 1:length(index_c)-1
                        cstart = index_c(block);
                        cend = index_c(block+1)-1;
                        psf1=gpuArray(single(squeeze(psf(:,:,u,v,cstart:cend  ))));
                        sumupXG=sumupXG+forwardProjectGPU(psf1, Xguess(:,:,cstart:cend));
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % key step
                    for uu=1:Nbx
                        for vv=1:Nby
                            sub_HXguess=sumupXG(borderx+(uu-1)*N3+1:borderx+uu*N3,bordery+(vv-1)*N3+1:bordery+vv*N3);
                            sub_blur_image=gpuArray(squeeze(blur_image(borderx+(uu-1)*N3+1-0:borderx+uu*N3+0,bordery+(vv-1)*N3+1-0:bordery+vv*N3+0,u,v)));
                            corr_map=gather(normxcorr2(sub_HXguess,sub_blur_image));
                            % figure(1)
                            % imshow(XxNorm(gather(sub_HXguess)));
                            % figure(2)
                            % imshow(XxNorm(gather(sub_blur_image)));
%                             figure(3)
%                             imshow(XxNorm(corr_map));
%                             if write_first_slice
%                                 sub_HXguess_list(:,:,1) = gather(sub_HXguess);
%                                 sub_blur_image_list(:,:,1) = gather(sub_blur_image);
%                                 corr_map_list(:,:,1) = corr_map;
%                                 write_first_slice = 0;
%                             else
%                                 sub_HXguess_list(:,:,end+1) = gather(sub_HXguess);
%                                 sub_blur_image_list(:,:,end+1) = gather(sub_blur_image);
%                                 corr_map_list(:,:,end+1) = corr_map;
%                             end
%                             corr_map=gather(imregister(sub_HXguess,sub_blur_image));
                            [testa,testb]=find(corr_map==max(corr_map(:)));
                            map_wavshape(u,v,uu,vv,1)=testa-size(sub_blur_image,1)-sidelobe;
                            map_wavshape(u,v,uu,vv,2)=testb-size(sub_blur_image,2)-sidelobe;
                        end
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
        end
        % ascertain that the middle angle does not move
        for uu=1:Nbx
            for vv=1:Nby
                cx=map_wavshape(ceil(Nnum/2),ceil(Nnum/2),uu,vv,1);
                cy=map_wavshape(ceil(Nnum/2),ceil(Nnum/2),uu,vv,2);
                map_wavshape(:,:,uu,vv,1)=(squeeze(map_wavshape(:,:,uu,vv,1))-cx).*DAO_angles;
                map_wavshape(:,:,uu,vv,2)=(squeeze(map_wavshape(:,:,uu,vv,2))-cy).*DAO_angles;
            end
        end
        % limiting shift map to [-sidelobe,sidelobe]
        for uu=1:Nbx
            for vv=1:Nby
                for u=1:Nnum
                    for v=1:Nnum
                        map_wavshape(u,v,uu,vv,1)=min(max(map_wavshape(u,v,uu,vv,1).*map_wavshape_weight(u,v,uu,vv),-sidelobe),sidelobe);%+min(max(cx,-10),10);
                        map_wavshape(u,v,uu,vv,2)=min(max(map_wavshape(u,v,uu,vv,2).*map_wavshape_weight(u,v,uu,vv),-sidelobe),sidelobe);%+min(max(cx,-10),10);
                    end
                end
            end
        end

        % defocus
        if rm_defocus
            for uu=1:Nbx
                for vv=1:Nby
                    k1 = Sy.*squeeze(map_wavshape(:,:,uu,vv,1)).*DAO_angles+Sx.*squeeze(map_wavshape(:,:,uu,vv,2)).*DAO_angles;
                    k2 = Sx.*Sx+Sy.*Sy;
                    k=sum(k1(:))/sum(k2(:));
                    map_wavshape(:,:,uu,vv,1)=squeeze(map_wavshape(:,:,uu,vv,1))-k*Sy;
                    map_wavshape(:,:,uu,vv,2)=squeeze(map_wavshape(:,:,uu,vv,2))-k*Sx;
                    for u=1:Nnum
                        for v=1:Nnum
                            map_wavshape(u,v,uu,vv,1)=min(max(map_wavshape(u,v,uu,vv,1).*map_wavshape_weight(u,v,uu,vv),-sidelobe),sidelobe);%+min(max(cx,-10),10);
                            map_wavshape(u,v,uu,vv,2)=min(max(map_wavshape(u,v,uu,vv,2).*map_wavshape_weight(u,v,uu,vv),-sidelobe),sidelobe);%+min(max(cx,-10),10);
                        end
                    end
                end
            end
        end

        if rm_defocus
            mkdir(strcat(savepath,'/map_wavshape'));
            save([savepath,'/map_wavshape/iter_',num2str(i),'.mat'],'map_wavshape');
        else
            mkdir(strcat(savepath,'/map_wavshape_withDefocus'));
            save([savepath,'/map_wavshape_withDefocus/iter_',num2str(i),'.mat'],'map_wavshape');
        end
        temp1=zeros(Nbx*Nnum,Nby*Nnum,2);
        for uu=1:Nbx
            for vv=1:Nby
                temp1((uu-1)*Nnum+1:uu*Nnum,(vv-1)*Nnum+1:vv*Nnum,:)=squeeze(map_wavshape(:,:,uu,vv,:));
            end
        end
%         figure(10);subplot(121);imshow(temp1(:,:,1),[]);subplot(122);imshow(temp1(:,:,2),[]);
%         imwriteTFSK(sub_blur_image_list,strcat(savepath,'_iter',num2str(i),'_blur_image.tif'));
%         imwriteTFSK(sub_HXguess_list,strcat(savepath,'_iter',num2str(i),'_HXguess.tif'));
%         imwriteTFSK(single(corr_map_list),strcat(savepath,'_iter',num2str(i),'_corrmap.tif'));
%         sub_HXguess_list = zeros(1989,1989,1);
%         sub_blur_image_list = zeros(1989,1989,1);
%         corr_map_list = zeros(3977,3977,1);
%         write_first_slice = 1;
    end
    
    for u_2=1:Nnum
        for v_2=1:Nnum
            
            u=wjm_map_u((u_2-1)*Nnum+v_2);
            v=wjm_map_v((u_2-1)*Nnum+v_2);

            
            if (AOstar>0 && DAO_angles(u,v)==0) || (AOstar<=0 && recon_weight(u,v)==0)
                continue;
            else
                if AOstar>0 && i>=2                   
                    map_wavshape_x=squeeze(map_wavshape(u,v,:,:,1));
                    map_wavshape_y=squeeze(map_wavshape(u,v,:,:,2));
                    map_wavshape_x1=imresize(map_wavshape_x,[round(img_r/3),round(img_c/3)],'nearest');
                    map_wavshape_y1=imresize(map_wavshape_y,[round(img_r/3),round(img_c/3)],'nearest');
                    map_wavshape_xx=imresize(map_wavshape_x1,[img_r,img_c],'cubic');
                    map_wavshape_yy=imresize(map_wavshape_y1,[img_r,img_c],'cubic');
                    blur_image_uv=gpuArray(interp2(coordinate1,coordinate2,blur_image(:,:,u,v),coordinate1+map_wavshape_yy,coordinate2+map_wavshape_xx,'cubic',0));
                else
                    blur_image_uv=gpuArray(blur_image(:,:,u,v));
                end
                sumupXG = gpuArray.zeros(img_r,img_c,'single');
                for block = 1:length(index_c)-1
                    cstart = index_c(block);
                    cend = index_c(block+1)-1;
                    psf1=gpuArray(single(squeeze(psf(:,:,u,v,cstart:cend  ))));
                    sumupXG=sumupXG+forwardProjectGPU(psf1, Xguess(:,:,cstart:cend));
                end     
                errorEM=blur_image_uv./sumupXG;
                errorEM(~isfinite(errorEM))=0;
                XguessCor = gpuArray.zeros(img_r,img_c,allz,'single');
                for block = 1:length(index_c)-1
                    cstart = index_c(block);
                    cend = index_c(block+1)-1;
                    psf1=gpuArray(single(squeeze(psf(:,:,u,v,cstart:cend  ))));
                    XguessCor(:,:,cstart:cend)=backwardProjectGPU(psf1, errorEM);
                end
                HTF1 = gpuArray.zeros(img_r,img_c,allz,'single');
                for block = 1:length(index_c)-1
                    cstart = index_c(block);
                    cend = index_c(block+1)-1;
                    psf1=gpuArray(single(squeeze(psf(:,:,u,v,cstart:cend  ))));
                    HTF1(:,:,cstart:cend)=backwardProjectGPU(psf1, gpuArray(single(ones( size(Xguess,1),size(Xguess,2) ))));
                end
                
                XguessCor=Xguess.*XguessCor./ HTF1;
                clear psf1;
                clear HTF1;
                XguessCor(find(isnan(XguessCor))) = 0;
                XguessCor(find(isinf(XguessCor))) = 0;
                Xguess=XguessCor.*recon_weight(u,v)*1+(1-recon_weight(u,v)*1).*Xguess;
                Xguess(Xguess<thresh) = 0;
                Xguess(isnan(Xguess)) = 0;
            end
        end
    end
    
    if mod(i,save_iter)==0
        % imwriteTFSK(single(gather(Xguess)),strcat(savepath,'/iter',num2str(i),'_xcorr_thresh',num2str(thresh),'.tif'));
        if rm_defocus
            % imwriteTFSK(uint16(65535*XxNorm(gather(Xguess),min_prc,max_prc)),strcat(savepath,'_iter',num2str(i),'_xcorr_thresh',num2str(thresh),'.tif'));
            imwriteTFSK(gather(Xguess),strcat(savepath,'_iter',num2str(i),'_xcorr_thresh',num2str(thresh),'.tif'));
        else
            imwriteTFSK(uint16(65535*XxNorm(gather(Xguess),min_prc,max_prc)),strcat(savepath,'_withDefocus_iter',num2str(i),'_xcorr_thresh',num2str(thresh),'.tif'));
        end
    end
end

Xguess=gather(Xguess);
end

