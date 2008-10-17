global as;
as=20;
sz=71;
[I,D] = rotateAll({'../../data/JamesOrth/MCF7/MCF7 JDO76_S22_T023'}, ...
                  {'../../data/JamesOrth/MCF7/MCF7 JDO76_S22_T023.deriv'});

plot_no = 360/as;
for ai=1:plot_no
    Ibox = getRotatedBox(I,-57,-173,sz,(ai-1)*as);
    Dbox = getRotatedDerivBox(D,-57,-173,sz,(ai-1)*as);
    
    subplot(1,2,1);
    title([num2str((ai-1)*as) ' degrees']);
    imshow(Ibox);
    
    clear HSV
    HSV(1:sz,1:sz,3)=1;
    HSV(1:sz,1:sz,2)=Dbox.mag;
    HSV(1:sz,1:sz,1)=Dbox.angle;
    subplot(1,2,2);
    imshow(hsv2rgb(HSV));
    input('next? ');
end
