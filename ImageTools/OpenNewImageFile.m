function handles = OpenNewImageFile(handles)

% Help for the Open New Image File tool:
% Category: Image Tools
%
% Use this tool to open an image and display it (for example, in order
% to check the background pixel intensity or to determine pixel
% coordinates, or to check which wavelength it is).
%
% See also SHOWPIXELDATA, SHOWDATAONIMAGE.

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

cd(handles.Current.DefaultImageDirectory)
%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be shown.
[FileName,Pathname] = uigetfile('*.*','Select the image to view');
%%% If the user presses "Cancel", the FileName will = 0 and nothing will
%%% happen.
if FileName == 0
else
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
%%% SHOULD CONSIDER ADDING IT BACK.
%     %%% Acquires basic screen info for making buttons in the
%     %%% display window.
%     StdUnit = 'point';
%     StdColor = get(0,'DefaultUIcontrolBackgroundColor');
%     PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
    
    %%% Reads the image.
    Image = CPimread(fullfile(Pathname, FileName));
    figure; imagesc(Image), colormap(gray)
    pixval
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
%%% SHOULD CONSIDER ADDING IT BACK.
%     %%% The following adds the Interactive Zoom button, which relies
%     %%% on the InteractiveZoomSubfunction.m being in the CellProfiler
%     %%% folder.
%     set(FigureHandle, 'Unit',StdUnit)
%     FigurePosition = get(FigureHandle, 'Position');
%     %%% Specifies the function that will be run when the zoom button is
%     %%% pressed.
%     ZoomButtonCallback = 'try, InteractiveZoomSubfunction, catch CPmsgbox(''Could not find the file called InteractiveZoomSubfunction.m which should be located in the CellProfiler folder.''), end';
%     uicontrol('Parent',FigureHandle, ...
%         'CallBack',ZoomButtonCallback, ...
%         'BackgroundColor',StdColor, ...
%         'Position',PointsPerPixel*[FigurePosition(3)-108 5 105 22], ...
%         'String','Interactive Zoom', ...
%         'Style','pushbutton');
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
end
cd(handles.Current.StartupDirectory)