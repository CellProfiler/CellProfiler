function ShowPixelIntensityHistogram(handles)

% Help for the Show Pixel Intensity Histogram tool:
% Category: Image Tools
%
%
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

MsgboxHandle = CPmsgbox('Click on an image to generate a histogram of its pixel intensities. This window will be closed automatically - do not close it or click OK.');
%%% TODO: Should allow canceling.
waitforbuttonpress
ClickedImage = getimage(gca);
try ClickedFigureTitle = get(get(gca,'parent'),'name');
    if isempty(ClickedFigureTitle)
        try ClickedFigureTitle = ['Figure ',num2str(get(gca,'parent'))];
        catch ClickedFigureTitle = [];
        end
    end
catch
    try ClickedFigureTitle = ['Figure ',num2str(get(gca,'parent'))];
    catch ClickedFigureTitle = [];
    end
end
try ClickedImageTitle = get(get(gca,'title'),'string');
catch ClickedImageTitle = [];
end

if isempty(ClickedFigureTitle) ~= 1 | isempty(ClickedImageTitle) ~= 1
    Title = [ClickedFigureTitle,' ',ClickedImageTitle];
else Title =[];
end
try
    delete(MsgboxHandle)
end
drawnow
CPfigure(handles)
hist(ClickedImage(:),min(200,round(length(ClickedImage(:))/150)));
title(['Histogram for ' Title])
grid on