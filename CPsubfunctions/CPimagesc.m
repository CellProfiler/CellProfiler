function h = CPimagesc(Image,handles)

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

%%% Displays the image.
h = imagesc(Image);
%%% Embeds the Image tool submenu so that it appears when the user clicks on the image. 
set(h,'ButtonDownFcn','CPimagetool');

%%% Sets the user's preference for font size, which should affect tick
%%% labels and current and future titles.
set(gca,'fontsize',handles.Preferences.FontSize)

%%% Applies the user's choice for colormap.
if ndims(Image) == 2
    colormap(handles.Preferences.IntensityColorMap);
end