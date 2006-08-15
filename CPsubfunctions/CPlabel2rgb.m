function [im, handles]=CPlabel2rgb(handles, image)

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
numregion = double(max(image(:)));
if sum(sum(image)) >= 1
    if nargin == 2
        cmap = eval([handles.Preferences.LabelColorMap '(255)']);
        try
            if numregion>length(handles.newcmap)
                newregions = numregions-length(handles.newcmap);
                newindex = round(rand(1,newregions)*255);
                index = [index newindex];
                handles.newcmap = cmap(index,:,:);
            end
        catch
            S = rand('state');
            rand('state', 0);
            index = round(rand(1,numregion)*255);
            handles.Pipeline.TrackObjects.Colormap = cmap(index,:,:);
            rand('state', S);
        end
        im = label2rgb(image, handles.Pipeline.TrackObjects.Colormap, 'k', 'noshuffle');
    else
        cmap = eval([handles.Preferences.LabelColorMap '(max(2,max(image(:))))']);
        im = label2rgb(image, cmap, 'k', 'shuffle');
    end
else
    im=image;
end