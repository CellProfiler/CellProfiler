function ShowOrHidePixelData(handles)

% Help for the Show or Hide Pixel Data tool:
% Category: Image Tools
%
% SHORT DESCRIPTION:
% Shows X,Y pixel location and intensity information in the figure window.
% *************************************************************************
%
% This tool shows the pixel intensity at each X,Y location as you hover 
% over points within an image. The pixels are displayed via a small box at 
% the lower left corner of the figure window. If the image is color (RGB), 
% three intensity values are shown: Red, Green, and Blue.
%
% Currently, it can also measure lengths if you click the mouse at a
% starting point and hold the button down while dragging, although this
% could also be done with the Measure Length tool, accessible by clicking
% on the image of interest and choosing Measure Length from the resulting
% Image Tool window.
%
% To exit the tool, click the 'x' in the pixel intensity information panel.

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

%%% We want to use impixelinfoval (see below), but it fails when there are
%%% four subplots in a figure (e.g. GrayToColor). So for now we are using
%%% the outdated pixval command. The bug has been reported to Mathworks.
warning off Images:pixval:obsoleteFunction
pixval
warning on Images:pixval:obsoleteFunction

%%% If we ever do end up using impixelinfoval instead of pixval, we should
%%% also create a mechanism to delete the pixel info tool. Pixval works
%%% like a toggle so running this tool above repeatedly toggles it on/off,
%%% whereas the impixelinfoval is not designed to work that way.

% AllImagesHandles = findobj(gcf,'type','image');
% OneOfTheImagesHandle = AllImagesHandles(1);
% PixelInfoHandle = impixelinfoval(gcf,OneOfTheImagesHandle);
% try
%     set(PixelInfoHandle,'fontsize',handles.Preferences.FontSize)
% end