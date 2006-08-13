function Image = CPretrieveimage(handles,ImageName,ModuleName,ColorFlag,ScaleFlag,SizeFlag)

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
% $Revision: 2802 $

%%% Fills in missing arguments, if necessary.
if nargin == 5
%%% CPretrieveimage(handles,ImageName,ModuleName,ColorFlag,ScaleFlag)
    SizeFlag = 0;
elseif nargin == 3
%%% CPretrieveimage(handles,ImageName,ModuleName)
    ColorFlag = 0;
    ScaleFlag = 0;
    SizeFlag = 0;
end

if ischar(ColorFlag)
    if strcmpi(ColorFlag,'MustBeColor')
        ColorFlag = 3;
    elseif strcmpi(ColorFlag,'MustBeGray')
        ColorFlag = 2;
    elseif strcmpi(ColorFlag,'DontCheckColor')
        ColorFlag = 0;
    else
        error('The value you have chosen for the colorflag is invalid.');
    end
end

if ischar(ScaleFlag)
    if strcmpi(ScaleFlag,'CheckScale')
        ScaleFlag = 1;
    elseif strcmpi(ScaleFlag,'DontCheckScale')
        ScaleFlag = 0;
    else
        error('The value you have chosen for the scaleflag is invalid.');
    end
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
    error(['Image processing was canceled in the ', ModuleName, ' module because CellProfiler could not find the input image. CellProfiler expected to find an image named "', ImageName, '", but that image has not been created by the pipeline. Please adjust your pipeline to produce the image "', ImageName, '" prior to this ', ModuleName, ' module.'])
end
%%% Reads the image.
Image = handles.Pipeline.(ImageName);

if ScaleFlag == 1
    if max(Image(:)) > 1 || min(Image(:)) < 0
        CPwarndlg(['The image loaded in the ', ModuleName, ' module is outside the 0-1 range, and you may be losing data.'],'Outside 0-1 Range','replace');
    end
end

if ColorFlag == 2
    if ndims(Image) ~= 2
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement. This may be because the image is a color image.']);
    end
elseif ColorFlag == 3
    if ndims(Image) ~= 3
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is color, but the image loaded does not fit this requirement. This may be because the image is grayscale.']);
    end
end

if SizeFlag ~= 0
    %%% The try is necessary because if either image does not have the
    %%% proper number of dimensions, things will fail otherwise. If one of
    %%% the images (the SizeFlag or the Image itself) is 3-D (color), then
    %%% only the X Y dimensions are checked for size.
    try if any(SizeFlag(1:2) ~= size(Image(:,:,1)))
            error(['Image processing was canceled in the ', ModuleName, ' module. The incoming images are not all of equal size.']);
        end
    catch error(['Image processing was canceled in the ', ModuleName, ' module. The incoming images are not all of equal size.']);
    end
end