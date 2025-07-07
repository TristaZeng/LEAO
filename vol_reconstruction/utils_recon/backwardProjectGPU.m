function Backprojection = backwardProjectGPU(psf_t,projection)
psf_t=rot90(psf_t,2);
Backprojection=gpuArray.zeros(size(projection,1),size(projection,2),size(psf_t,3),'single');
[ra ca]=size(projection);
[rb cb]=size(psf_t(:,:,1));
r = ra+rb-1;c=ca+cb-1; p1 = (r-ra)/2;p2=(c-ca)/2;
b1 = gpuArray.zeros(r,c,'single');
a1 = gpuArray.zeros(r,c,'single');
a1(1:ra,1:ca) = projection(:,:) ;
f_a1=fft2(a1(1:ra,1:ca),r,c);
for z=1:size(psf_t,3)  
    b1(1:rb,1:cb) = psf_t(:,:,z) ;
    clear con1;
    con1 = ifft2(f_a1.*fft2(b1(1:rb,1:cb),r,c));
    Backprojection(:,:,z) = real(con1(p1+1:r-p1,p2+1:c-p2));   
end
end