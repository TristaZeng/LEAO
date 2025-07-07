function projection = forwardProjectGPU(psf,Xguess)
%coder.gpu.kernelfun;
%psf_ = gpuArray(psf(:,:,u,v,:));
[ra, ca]=size(Xguess(:,:,1));
[rb, cb]=size(psf(:,:,1));
r = ra+rb-1; c=ca+cb-1; p1 = (r-ra)/2;p2=(c-ca)/2;
a1 = gpuArray.zeros(r,c,'single');
b1 = gpuArray.zeros(r,c,'single');
con1 = gpuArray.zeros(r,c,'single');
for z=1:size(Xguess,3)
    a1(1:ra,1:ca) = Xguess(:,:,z);
    b1(1:rb,1:cb) = psf(:,:,z);                   
    %clear con1;
    con1 = con1 + fft2(a1(1:ra,1:ca),r,c) .* fft2(b1(1:rb,1:cb),r,c);                                                                  
end
projection1 = real(ifft2(con1));
projection = projection1(p1+1:r-p1,p2+1:c-p2);
end

