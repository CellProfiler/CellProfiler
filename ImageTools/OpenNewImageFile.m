function OpenNewImageFile(handles)

% Help for the Open New Image File tool:
% Category: Image Tools
%
% Use this tool to open an image and display it (for example, in order
% to check the background pixel intensity or to determine pixel
% coordinates, or to check which wavelength it is).
%
% The Open New Image File button shows pixel values in the range 0 to 1.
% Images are loaded into CellProfiler in this range so that modules behave
% consistently. The display is set to the same range so that, for example, 
% if a user wants to look at an image in order to determine which threshold
% to use within a module, the pixel values are directly applicable.
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

%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be shown.
%%% Current directory temporarily changed to default image directory 
%%% for image selection and then immediately restored
TempCD=cd;
cd(handles.Current.DefaultImageDirectory);
[FileName,Pathname] = uigetfile({'*.bmp;*.BMP;*.cur;*.CUR;*.fts;*.FTS;*.fits;*.FITS;*.gif;*.GIF;*.hdf;*.HDF;*.ico;*.ICO;*.jpg;*.JPG;*.jpeg;*.JPEG;*.pbm;*.PBM;*.pcx;*.PCX;*.pgm;*.PGM;*.png;*.PNG;*.pnm;*.PNM;*.ppm;*.PPM;*.ras;*.RAS;*.tif;*.TIF;*.tiff;*.TIFF;*.xwd;*.XWD;*.dib;*.DIB', 'All Image Files';'*.*',  'All Files (*.*)'},'Select the image to view');
cd(TempCD);
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
    CPfigure(handles);
    imagesc(Image);
    colormap(gray);
    pixval;
    title(FileName);

    
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