function LoadedImage = CPimread(CurrentFileName)
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
% $Revision$

if nargin == 0 %returns the vaild image extensions
    formats = imformats;
    LoadedImage = [cat(2, formats.ext) {'dib'} {'mat'} {'fig'} {'zvi'}]; %LoadedImage is not a image here, but rather a set
    return
elseif nargin == 1,
    %%% Handles a non-Matlab readable file format.
    [Pathname, FileName, ext] = fileparts(char(CurrentFileName));
    if strcmp('.DIB', upper(ext)),
        %%% Opens this non-Matlab readable file format.  Compare to
        %%% Matlab Central file id 11096, which does the same thing.
	%%% The DIB (Device Independent Bitmap) format is a format %%
        %%% used (mostly internally) by MS Windows.  Cellomics has
        %%% adopted it for their instruments.
        fid = fopen(char(CurrentFileName), 'r');
        if (fid == -1),
            error(['The file ', char(CurrentFileName), ' could not be opened. CellProfiler attempted to open it in DIB file format.']);
        end
        A = fread(fid, 52, 'uint8=>uint8');
	HeaderLength = from_little_endian(A(1:4));
	if HeaderLength ~= 40
	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB files with header length other than %d', char(CurrentFileName), HeaderLength));
	end
        Width = from_little_endian(A(5:8));
        Height = from_little_endian(A(9:12));
        % All Cellomics DIB files we have seen have had a bit depth of
        % 16.  However, all instruments we know that use this file
        % format have 12-bit cameras, so we hard-code 12-bits.  This
        % may change in the future if we encounter images of a
        % different bit depth.
	BitDepth = from_little_endian(A(15:16));
	if BitDepth == 16
	  BitDepth = 12;
	else
	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB files with bit depth %d', char(CurrentFileName), BitDepth));
	end

        Channels = from_little_endian(A(13:14));
	Compression = from_little_endian(A(17:20));
	if Compression ~= 0
	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB compression of type %d', char(CurrentFileName), Compression));
	end
	% We have never seen a DIB file with more than one channel.
        % It seems reasonable to assume that the second channel would
        % follow the first, but this needs to be verified.
        LoadedImage = zeros(Height,Width,Channels);
	for c=1:Channels,
	  % The 'l' causes convertion from little-endian byte order.
	  [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
	  if Count < (Width * Height),
	    fclose(fid);
	    error(['End-of-file encountered while reading ', char(CurrentFileName), '. Have you entered the proper size and number of channels for these images?']);
	  end
	  LoadedImage(:,:,c) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
	end
        fclose(fid);
    elseif strcmp('.MAT',upper(ext))
        load(CurrentFileName);
        if exist('Image','var')
            LoadedImage = Image;
        else
            error('Was unable to load the image.  This could be because the .mat file specified is not a proper image file');
        end
    elseif strcmp('.ZVI',upper(ext))
        try
            %%% Read (open) the image you want to analyze and assign it to a variable,
            %%% "LoadedImage".
            %%% Opens Matlab-readable file formats.
            LoadedImage = im2double(CPimreadZVI(char(CurrentFileName)));
        catch
            error(['Image processing was canceled because the module could not load the image "', char(CurrentFileName), '" in directory "', pwd,'".  The error message was "', lasterr, '"'])
        end
    else
        try
            Header = imfinfo(CurrentFileName);
            if isfield(Header,'Model') & any(strfind(Header(1).Model,'GenePix'))
                PreLoadedImage = imreadGP([FileName,ext],Pathname);
                LoadedImage(:,:,1)=double(PreLoadedImage(:,:,1))/65535;
                LoadedImage(:,:,2)=double(PreLoadedImage(:,:,1))/65535;
                LoadedImage(:,:,3)=zeros(size(PreLoadedImage,1),size(PreLoadedImage,2));
            else
                %%% Read (open) the image you want to analyze and assign it to a variable,
                %%% "LoadedImage".
                %%% Opens Matlab-readable file formats.
                LoadedImage = im2double(imread(char(CurrentFileName)));
            end
        catch
            error(['Image processing was canceled because the module could not load the image "', char(CurrentFileName), '" in directory "', pwd,'".  The error message was "', lasterr, '"'])
        end
    end
end


function ImageArray = imreadZVI(CurrentFileName)

% Open .zvi file
fid = fopen(char(CurrentFileName), 'r');
if (fid == -1),
    error(['The file ', char(CurrentFileName), ' could not be opened. CellProfiler attempted to open it in ZVI file format.']);
end

%read and store data
[A, Count] = fread(fid, inf, 'uint8'  , 0, 'l');

fclose(fid);

%find first header block and returns the position of the header
indices = find(A == 65);
counter = 0;
for i = 1:length(indices)
    if A(indices(i)+1) == 0 && A(indices(i)+2) == 16
        counter = counter+1;
        block1(counter) = indices(i);
    end
end

%checks that file is in the proper format and finds another block,
%returns the positioni of where to read image information
for i = 1:length(block1)
    pos = block1(3)+22;
    if (A(pos) == 65 && A(pos+1) == 0 && A(pos+2) == 128) & A(pos+3:pos+13) == zeros(11,1)
        pos = pos+133;
        for i = pos:length(A)
            if A(i) == 32 && A(i+1) == 0 && A(i+2) == 16
                newpos = i+3;
                break;
            end
        end
        if exist('newpos')
            break;
        end
    end
end

%stores information in byte arrays
Width = A(newpos: newpos+3);
Height = A(newpos+4: newpos + 7);
BytesPixel = A(newpos+12:newpos+15);
PixelType = A(newpos+16:newpos+19);
newpos = newpos+24;

%Get decimal values of the width and height
Width = from_little_endian(Width);
Height = from_little_endian(Height);
BytesPixel = from_little_endian(BytesPixel);

%Finds and stores the data vector
NumPixels = Width*Height*BytesPixel;
ImageData = A(newpos:newpos+NumPixels-1);

%     %Might be useful to add on as needed.....
%     if PixelType == 1|8
%         %this is a grayscaled image
%     elseif PixelType == 3 | 4
%         %this is a color image
%     end

%Stores and returns Image Array
ImageArray=reshape(ImageData, Width, Height)';

% One and two and three little-endians...
function i = from_little_endian(byte_array)
is_little_endian = typecast(uint8([1 0]), 'uint16') == 1;
if size(byte_array,2) == 1
  byte_array = byte_array';
end
switch size(byte_array,2)
 case 2
  type = 'uint16';
 case 4
  type = 'uint32';
 otherwise
  error('Don''t know what to do with a byte array of this length.')
end
if is_little_endian
  i = double(typecast(byte_array, type));
else
  i = double(swapbytes(typecast(byte_array, type)));
end

function [Image Header] = imreadGP(filename,filedir)
% IMREADGP
%      imreadGP reads in a an image file from GenePix
%      imreadGP ignores the preview images and collects the header
%      information
%
%      [Image Header] = imreadGP(filename,filedir)
%
%      Image is a M by N by Idx matrix where Idx is the number of full size
%      images in the multi-image tiff file
%      Header is a structure returned from imfinfo where information on
%      preview images has been removed.
%
%      Created by Hy Carrinski
%      Broad Institute

% QC part 1
if nargin < 2 || isempty(filedir)
    filedir = [];
elseif ~strcmp(filedir(end),'\') || ~strcmp(filedir(end),'/')
    filedir = [filedir filesep];
end

if nargin < 1 || isempty(filename)
    error('filename omitted');
end

% Read header info into a structure
Header = imfinfo([filedir filename]);

% QC part 2
if any(~isfield(Header,{'ImageDescription','Width','Height','BitDepth'}))
    error('Required Tiff fields are not defined');
end

% Parse header info to determine good images
ImDesc = strvcat(Header.ImageDescription);
imageIdx = find(ImDesc(:,6) == 'W'); % indices of "good" images
if numel(imageIdx) < size(ImDesc,1)
    display('Preview image removed');
end

% QC part 3
if numel(unique([Header(imageIdx).Width])) ~= 1 || ...
        numel(unique([Header(imageIdx).Height])) ~= 1
    error('Images are not all of the same size');
end

% Initialize image matrix
Image = zeros(Header(imageIdx(1)).Height,Header(imageIdx(1)).Width,'uint16');

% Read in images
for i = 1:numel(imageIdx)
    Image(:,:,i) = imread([filedir filename],imageIdx(i));
end

% Remove Header fields from "not good" images
Header(setdiff(1:numel(Header),imageIdx)) = [];

%num2str(Header(imageNums(1)).BitDepth) %possible later addition