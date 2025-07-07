function WDF=realign(sLF,Nshift,Nnum)
multiWDF=zeros(Nnum,Nnum,size(sLF,1)/Nnum,size(sLF,2)/Nnum,Nshift,Nshift); %% multiplexed phase-space
    for i=1:Nnum
        for j=1:Nnum
            for a=1:size(sLF,1)/Nnum
                for b=1:size(sLF,2)/Nnum
                    multiWDF(i,j,a,b,:,:)=squeeze(  sLF(  (a-1)*Nnum+i,(b-1)*Nnum+j,:,:  )  );
                end
            end
        end
    end
    WDF=zeros(  size(sLF,1)/Nnum*Nshift,size(sLF,2)/Nnum*Nshift,Nnum,Nnum  ); % multiplexed phase-space
    for a=1:size(sLF,1)/Nnum
        for c=1:Nshift
            x=Nshift*a+1-c;
            for b=1:size(sLF,2)/Nnum
                for d=1:Nshift
                    y=Nshift*b+1-d;
                    WDF(x,y,:,:)=squeeze(multiWDF(:,:,a,b,c,d));
                end
            end
        end
    end
end