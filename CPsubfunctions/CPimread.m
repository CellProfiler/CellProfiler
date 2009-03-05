function LoadedImage = CPimread(CurrentFileName, flex_idx)
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% CPimread by itself returns the vaild image extensions
% CPimread(CurrentFileName) is used for most filetypes
% CPimread(CurrentFileName, flex_idx) is used for 'tif,tiff,flex movies'
% option
%       in LoadImages, where flex_idx is index of a particular image within the
%       file
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

if nargin == 0 %returns the vaild image extensions
    formats = imformats;
    LoadedImage = [cat(2, formats.ext) {'dib'} {'mat'} {'fig'} {'zvi'} {'raw'},{'flex'}]; %LoadedImage is not a image here, but rather a set
    return
elseif nargin == 1,
    % The following lines make sure that all directory separation
    % characters follow the platform format
    CurrentFileName = char(CurrentFileName);
    CurrentFileName = strrep(strrep(CurrentFileName,'\',filesep),'/',filesep);
    
    %%% Handles a non-Matlab readable file format.
    [Pathname, FileName, ext] = fileparts(CurrentFileName);
    
    
    
    if strcmp('.DIB', upper(ext)),
        %%% Opens this non-Matlab readable file format.  Compare to
        %%% Matlab Central file id 11096, which does the same thing.
	%%% The DIB (Device Independent Bitmap) format is a format %%
        %%% used (mostly internally) by MS Windows.  Cellomics has
        %%% adopted it for their instruments.
        fid = fopen(CurrentFileName, 'r');
        if (fid == -1),
            error(['The file ', CurrentFileName, ' could not be opened. CellProfiler attempted to open it in DIB file format.']);
        end
        A = fread(fid, 52, 'uint8=>uint8');
	HeaderLength = from_little_endian(A(1:4));
	if HeaderLength ~= 40
	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB files with header length %d', CurrentFileName, HeaderLength));
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
	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB files with bit depth %d', CurrentFileName, BitDepth));
	end

        Channels = from_little_endian(A(13:14));
	Compression = from_little_endian(A(17:20));
	% Commenting this line out 03/05/09, since images with Compression ~= 0
	% seem to be OK
%     if Compression ~= 0
% 	  error(sprintf('The file %s could not be opened because CellProfiler does not understand DIB compression of type %d', CurrentFileName, Compression));
% 	end
	% We have never seen a DIB file with more than one channel.
    % It seems reasonable to assume that the second channel would
    % follow the first, but this needs to be verified.
    LoadedImage = zeros(Height,Width,Channels);
	for c=1:Channels,
	  % The 'l' causes convertion from little-endian byte order.
	  [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
	  if Count < (Width * Height),
	    fclose(fid);
	    error(['End-of-file encountered while reading ', CurrentFileName, '. Have you entered the proper size and number of channels for these images?']);
	  end
	  LoadedImage(:,:,c) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
	end
        fclose(fid);
    elseif strcmp('.RAW',upper(ext)),
        fid = fopen(CurrentFileName,'r');
        if (fid == -1),
            error(['The file ', CurrentFileName, ' could not be opened. CellProfiler attempted to open it in RAW file format.']);
        end
        A = fread(fid,52,'uint8=>uint8');
        HeaderLength = from_little_endian(A(1:4));
        Width = from_little_endian(A(17:20));
        Height = from_little_endian(A(21:24));
        BitDepth = from_little_endian(A(25:28));
        LoadedImage = zeros(Height,Width);
        %
        % Skip the rest of the header
        %
        [Data, Count] = fread(fid, HeaderLength-52,'uint8',0,'l');
	
	  % The 'l' causes convertion from little-endian byte order.
	  [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
	  if Count < (Width * Height),
	    fclose(fid);
	    error(['End-of-file encountered while reading ', CurrentFileName, '. Have you entered the proper size and number of channels for these images?']);
	  end
	  LoadedImage(:,:,1) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
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
            % Read (open) the image you want to analyze and assign it to a variable,
            % "LoadedImage".
            % Opens Matlab-readable file formats.
            LoadedImage = im2double(CPimreadZVI(CurrentFileName));
        catch
            error(['Image processing was canceled because the module could not load the image "', CurrentFileName, '" in directory "', pwd,'".  The error message was "', lasterr, '"'])
        end
    elseif strcmp('.FLEX',upper(ext))
        CPwarndlg('Flex files support is still under development.  The image displayed is likely only the first image within the file')
        % TODO: Display subplots of all images within one flex file (can
        % happen when image double-clicked in main GUI)
        % For now, we will just disaply the first image...
        LoadedImage = im2double(imread(CurrentFileName));
    else
        try
            if IsGenePix(CurrentFileName)
                PreLoadedImage = CPimreadGP([FileName,ext],Pathname);
                LoadedImage(:,:,1)=double(PreLoadedImage(:,:,1))/65535;
                LoadedImage(:,:,2)=double(PreLoadedImage(:,:,1))/65535;
                LoadedImage(:,:,3)=zeros(size(PreLoadedImage,1),size(PreLoadedImage,2));
            else
                %%% Read (open) the image you want to analyze and assign it to a variable,
                %%% "LoadedImage".
                %%% Opens Matlab-readable file formats.
                LoadedImage = im2double(imread(CurrentFileName));
            end
        catch
            error(['Image processing was canceled because the module could not load the image "', CurrentFileName, '" in directory "', pwd,'".  The error message was "', lasterr, '"'])
        end
    end
elseif nargin == 2,  %% Only used for 'tif,tiff,flex movies' option in LoadImages
    LoadedImage = CPimread_flex(CurrentFileName, flex_idx);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ImageArray = CPimreadZVI(CurrentFileName)

% Open .zvi file
fid = fopen(CurrentFileName, 'r');
if (fid == -1),
    error(['The file ', CurrentFileName, ' could not be opened. CellProfiler attempted to open it in ZVI file format.']);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Image Header] = CPimreadGP(filename,filedir)
% CPIMREADGP
%      CPimreadGP reads in a an image file from GenePix
%      CPimreadGP ignores the preview images and collects the header
%      information
%
%      [Image Header] = CPimreadGP(filename,filedir)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ScaledImage = CPimread_flex(imname, flex_idx)
%%% Read a .flex file, with possible scaling information.  No
%%% documentation is available, as far as I know.  Most of what is
%%% below is based on experiments, looking at the plaintext XML in the
%%% .flex files, and by reference to FlexReader.java in the LOCI
%%% Bio-Formats project.  


% First load the image...
RawImage = imread(imname, flex_idx);

% Then go back and try to get the scaling factors...
try
    % get the file contents
    fid = fopen(imname, 'r');
    FileAsString = fread(fid, inf, 'uint8=>char')';
    fclose(fid);

    % Extract the Array information
    strstart = findstr('<Arrays>', FileAsString);
    strend = findstr('</Arrays>', FileAsString);
    ArrayString = FileAsString(strstart(1):strend(1));

    % Find the Factors
    factor_locations = findstr('Factor="', ArrayString);
    
    % determine maximum factor value, as it decides the number of bits to convert to, below.
    for i = 1:length(factor_locations),
        % get the ith factor string, and extract the value
        IdxFactorStringStart = ArrayString(factor_locations(i) + length('Factor="'):end);
        strend = findstr(IdxFactorStringStart, '"') - 1;
        IdxFactorString = IdxFactorStringStart(1:strend);
        ScalingFactors(i) = str2double(IdxFactorString);
    end

    %%% The logic here mirrors that in FlexReader.java, part of the LOCI Bio-Formats package
    %%% Note: We had special considerations for 8- vs. 16-bit images, but
    %%% this was deemed unnecessary
    if max(ScalingFactors) > 256,
        % upgrade to 32 bits
        ScaledImage = uint32(RawImage) * ScalingFactors(flex_idx);
    else
        % upgrade to 16 bits (if 8-bit)
        ScaledImage = uint16(RawImage) * ScalingFactors(flex_idx);
    end

catch
    % default is no scaling (for non-flex tiffs)
    ScaledImage = RawImage;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function IsGenePix - determine whether a file is a GenePix TIFF file
%
% The TIFF file format specification can be found at
% http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
%
function is_gene_pix = IsGenePix(filename)
is_gene_pix = 0;
fid = fopen(filename,'r');
try
    % Bytes 0-1 of the file specify the byte order
    % Byte order is either 'II' for least significant
    % or 'MM' for most significant
    ByteOrder = reshape(fread(fid,2,'uint8=>char'),1,2);
    if strcmp(ByteOrder,'II')
        ByteOrder = 'ieee-le';
    elseif strcmp(ByteOrder,'MM')
        ByteOrder = 'ieee-be';
    else
        fclose(fid);
        return
    end
    % Bytes 2-3 identify the file as a TIFF file. They always
    % have the value 42.
    TiffCookie = fread(fid,1,'int16',0,ByteOrder);
    if TiffCookie ~= 42
        fclose(fid);
        return
    end
    % Bytes 4-7 have an offset to the first IFD
    Offset = fread(fid,1,'uint32',0,ByteOrder);
    if fseek(fid,Offset,'bof') ~= 0
        fclose(fid);
        return
    end
    % An IFD has a 2-byte record count
    RecordCount = fread(fid,1,'uint16',0,ByteOrder);
    for i=1:RecordCount
        Tag = fread(fid,1,'uint16',0,ByteOrder);
        Type = fread(fid,1,'uint16',0,ByteOrder);
        Count = fread(fid,1,'uint32',0,ByteOrder);
        Offset = fread(fid,1,'uint32',0,ByteOrder);
        %
        % GenePix files are identified as having a model of GenePix
        % The model tag is # 272
        % A type of 2 is ASCII
        %
        if Tag == 272 && Type == 2
            if fseek(fid,Offset,'bof') ~= 0
                fclose(fid);
                return
            end
            % ASCII strings are null-terminated, so we don't
            % read the null (Count-1)
            Model = reshape(fread(fid,Count-1,'uint8=>char'),1,Count-1);
            if strcmp(Model,'GenePix')
                is_gene_pix = 1;
            end
            fclose(fid);
            return
        end
    end
    fclose(fid);
catch
    fclose(fid);
    return
end