function image = CPretrieveimage(handles,ImageName,ModuleName,ColorFlag,ScaleFlag,SizeFlag)

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

if nargin == 5
    SizeFlag = 0;
elseif nargin == 3
    ColorFlag = 0;
    ScaleFlag = 0;
    SizeFlag = 0;
end

%%% Checks whether the image to be analyzed exists in the handles
%%% structure.
if ~isfield(handles.Pipeline, ImageName)
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because the input image could not be found.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
image = handles.Pipeline.(ImageName);

if ScaleFlag == 1
    if max(image(:)) > 1 || min(image(:)) < 0
        CPwarndlg(['The first image that you loaded in the ', ModuleName, ' module is outside the 0-1 range, and you may be losing data.'],'Outside 0-1 Range','replace');
    end
end

if ColorFlag == 2
    if ndims(image) ~= 2
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.']);
    end
elseif ColorFlag == 3
    if ndims(image) ~= 3
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is color, but the image loaded does not fit this requirement.  This may be because the image is grayscale.']);
    end
end

if SizeFlag ~= 0
    if any(SizeFlag ~= size(image))
        error(['Image processing was canceled in the ', ModuleName, ' module. The incoming images are not all of equal size.']);
    end
end