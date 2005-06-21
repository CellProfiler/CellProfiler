function CPFixAspectRatio(OrigImage)
FigPos = get(gcf,'Position');
SubPos = size(OrigImage)
ScSize = get(0,'ScreenSize');

if(FigPos(3)/FigPos(4)) > (SubPos(2)/SubPos(1))
    FigPos(4) = FigPos(3) * SubPos(1) / SubPos(2);
else
    FigPos(3) = FigPos(4) * SubPos(2) / SubPos(1);
end

HorRatio = FigPos(3) / ScSize(3);
VerRatio = FigPos(4) / ScSize(4);
if(HorRatio > .8) || (VerRatio > .8)
    if(HorRatio > VerRatio)
        FigPos(3) = ScSize(3) * 0.8;
        FigPos(4) = FigPos(4) / FigPos(3) * ScSize(3) * 0.8;
    else
        FigPos(4) = ScSize(4) * 0.8;
        FigPos(3) = FigPos(3) / FigPos(4) * ScSize(4) * 0.8;
    end
end        

set(gcf,'Position',FigPos);