function CPFixAspectRatio(OrigImage)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

FigPosOrig = get(gcf,'Position');
FigPos = get(gcf,'Position');
SubPos = size(OrigImage);
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

if FigPos(4) > FigPosOrig(4)
    FigPos(2) = ScSize(4) - (FigPos(4)+80);
    set(gcf,'Position',FigPos);
else
    set(gcf,'Position',FigPos);
end