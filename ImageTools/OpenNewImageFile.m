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
%   Anne Carpenter
%   Thouis Jones
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

%%% Opens a user interface window which retrieves a file name and path
%%% name for the image to be shown.
%%% Current directory temporarily changed to default image directory
%%% for image selection and then immediately restored

ListOfExtensions = CPimread;
ImageExtCat = ['*.' ListOfExtensions{1}];
for i = 2:length(ListOfExtensions)
    ImageExtCat = [ImageExtCat ';*.' ListOfExtensions{i}];
end

TempCD=CPcd;
CPcd(handles.Current.DefaultImageDirectory);
[FileName,Pathname] = uigetfile({ImageExtCat, 'All Image Files';'*.*',  'All Files (*.*)'},'Select the image to view');
CPcd(TempCD);
%%% If the user presses "Cancel", the FileName will = 0 and nothing will
%%% happen.
if FileName == 0
else
    %%% Reads the image.
    Image = CPimread(fullfile(Pathname, FileName));
    CPfigure(handles);
    imagesc(Image);
    colormap(gray);
    FileName = strrep(FileName,'_','\_');
    title(FileName);
end