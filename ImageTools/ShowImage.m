function handles = ShowImage(handles)

% Help for the Show Image tool:
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
    Image = imreadimagefile(Pathname, FileName);
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
%     ZoomButtonCallback = 'try, InteractiveZoomSubfunction, catch msgbox(''Could not find the file called InteractiveZoomSubfunction.m which should be located in the CellProfiler folder.''), end';
%     uicontrol('Parent',FigureHandle, ...
%         'CallBack',ZoomButtonCallback, ...
%         'BackgroundColor',StdColor, ...
%         'Position',PointsPerPixel*[FigurePosition(3)-108 5 105 22], ...
%         'String','Interactive Zoom', ...
%         'Style','pushbutton');
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
end
cd(handles.Current.StartupDirectory)

%%%%%%%%%%%%%%%%%%%%%
%%%% SUBFUNCTION %%%%
%%%%%%%%%%%%%%%%%%%%%
function LoadedImage = imreadimagefile(Pathname, FileName)
%%% Check extension of FileName
[pathstr, name, ext] = fileparts(FileName);
if strcmp('.DIB', upper(ext)),
    %%% Opens this non-Matlab readable file format.
    Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
    Width = str2double(Answers{1});
    Height = str2double(Answers{2});
    BitDepth = str2double(Answers{3});
    Channels = str2double(Answers{4});
    fid = fopen(fullfile(Pathname, FileName), 'r');
    if (fid == -1),
        error(['The file ', char(CurrentFileName), ' could not be opened. CellProfiler attempted to open it in DIB file format.']);
    end
    fread(fid, 52, 'uchar');
    LoadedImage = zeros(Height,Width,Channels);
    for c=1:Channels,
        [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
        if Count < (Width * Height),
            fclose(fid);
            error(['End-of-file encountered while reading ', char(CurrentFileName), '. Have you entered the proper size and number of channels for these images?']);
        end
        LoadedImage(:,:,c) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
    end
    fclose(fid);
else
    %%% Opens Matlab-readable file formats.
    try
        %%% Read (open) the image you want to analyze and assign it to a variable,
        %%% "LoadedImage".
        LoadedImage = im2double(imread(fullfile(Pathname, FileName)));
    catch error(['Unable to open the file, perhaps not a valid Image File.']);
    end
end