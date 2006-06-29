function OpenNewImageFile(handles)

% Help for the Open New Image File tool:
% Category: Image Tools
%
% SHORT DESCRIPTION:
% Opens an image file in a new window.
% *************************************************************************
%
% Use this tool to open an image and display it. Images are loaded into
% CellProfiler in the range of 0 to 1 so that modules behave consistently.
% The display is contrast stretched so that the brightest pixel in the
% image is white and the darkest is black for easier viewing.

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
    try
        %%% Reads the image.
        Image = CPimread(fullfile(Pathname, FileName));
    catch CPerrordlg(lasterr)
        return
    end
    %%% Opens a new figure window.
    FigureHandle = figure;
    if ~isfield(handles,'Pipeline')
        handles.Pipeline = [];
    end
    CPfigure(handles,'Image',FigureHandle);
    CPimagesc(Image,handles);
    FileName = strrep(FileName,'_','\_');
    title(FileName);
end