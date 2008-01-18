function [im, handles]=CPlabel2rgb(handles, image)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(image)) >= 1
    cmap = eval([handles.Preferences.LabelColorMap '(max(2,max(image(:))))']);
    im = label2rgb(image, cmap, 'k', 'shuffle');
else
    im=image;
end



