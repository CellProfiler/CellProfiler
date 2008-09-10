function [S, stack_cnt] = CPtiffread(filename, img_first, img_last)
% [S, nbimages] = tiffread;
% [S, nbimages] = tiffread(filename);
% [S, nbimages] = tiffread(filename, image);
% [S, nbimages] = tiffread(filename, first_image, last_image);
%
% Reads 8,16,32 bits uncompressed grayscale tiff and stacks of tiff images,
% for example those produced by metamorph or NIH-image. The function can be
% called with a file name in the current directory, or without argument, in
% which case it pop up a file openning dialog to allow selection of file.
% the pictures read in a stack can be restricted by specifying the first
% and last images to read, or just one image to read.
%
% at return, nbimages contains the number of images read, and S is a vector
% containing the different images with their tiff tags informations. The image
% pixels values are stored in the field .data, in the native format (integer),
% and must be converted to be used in most matlab functions.
%
% EX. to show image 5 read from a 16 bit stack, call image( double(S(5).data) );
%
% Francois Nedelec, EMBL, Copyright 1999-2003.
% last modified April 4, 2003.
% Please, feedback/bugs/improvements to  nedelec (at) embl.de
%
% Used with permission by CellProfiler.

% $Revision$

if (nargin == 0)
    [filename, pathname] = uigetfile('*.tif;*.stk', 'select image file');
    filename = [ pathname, filename ];
end

if (nargin<=1)
    img_first = 1;
    img_last = 10000;
end
if (nargin==2)
    img_last = img_first;
end

img_skip  = 0;
img_read  = 0;
stack_cnt = 1;

% not all valid tiff tags have been included, but they can be easily added
% to this code (see the official list of tags at
% http://partners.adobe.com/asn/developer/pdfs/tn/TIFF6.pdf
%
% the structure TIFIM is returned to the user, while TIF is not.
% so tags usefull to the user should be stored in TIFIM, while
% those used only internally can be stored in TIF.


% set defaults values :
TIF.sample_format     = 1;
TIF.samples_per_pixel = 1;
TIF.BOS               = 'l';          %byte order string

if  isempty(findstr(filename,'.'))
    filename=[filename,'.tif'];
end

TIF.file = fopen(filename,'r','l');
if TIF.file == -1
    filename = strrep(filename, '.tif', '.stk');
    TIF.file = fopen(filename,'r','l');
    if TIF.file == -1
        error(['file <',filename,'> not found.']);
    end
end

% read header
% read byte order: II = little endian, MM = big endian
byte_order = char(fread(TIF.file, 2, 'uchar')); %#ok Ignore MLint
if ( strcmp(byte_order', 'II') )
    TIF.BOS = 'l';                                %normal PC format
elseif ( strcmp(byte_order','MM') )
    TIF.BOS = 'b';
else
    error('This is not a TIFF file (no MM or II).');
end

%----- read in a number which identifies file as TIFF format
tiff_id = fread(TIF.file,1,'uint16', TIF.BOS);
if (tiff_id ~= 42)
    error('This is not a TIFF file (missing 42).');
end

%----- read the byte offset for the first image file directory (IFD)
ifd_pos = fread(TIF.file,1,'uint32', TIF.BOS);

while (ifd_pos ~= 0)

    clear TIFIM;
    TIFIM.filename = [ pwd, '\', filename ];
    % move in the file to the first IFD
    fseek(TIF.file, ifd_pos,-1);
    %disp(strcat('reading img at pos :',num2str(ifd_pos)));

    %read in the number of IFD entries
    num_entries = fread(TIF.file,1,'uint16', TIF.BOS);
    %disp(strcat('num_entries =', num2str(num_entries)));

    %read and process each IFD entry
    for i = 1:num_entries
        file_pos  = ftell(TIF.file);                     % save the current position in the file
        TIF.entry_tag = fread(TIF.file, 1, 'uint16', TIF.BOS);      % read entry tag
        entry = readIFDentry(TIF);
        %disp(strcat('reading entry <',num2str(entry_tag),'>:'));

        switch TIF.entry_tag
            case 254
                TIFIM.NewSubfiletype = entry.val;
            case 256         % image width - number of column
                TIFIM.width          = entry.val;
            case 257         % image height - number of row
                TIFIM.height         = entry.val;
            case 258         % bits per sample
                TIFIM.bits           = entry.val;
                TIF.bytes_per_pixel  = entry.val / 8;
                %disp(sprintf('%i bits per pixels', entry.val));
            case 259         % compression
                if (entry.val ~= 1)
                    error('Compression format not supported.');
                end
            case 262         % photometric interpretatio
                TIFIM.photo_type     = entry.val;
            case 269
                TIFIM.document_name  = entry.val;
            case 270         % comment:
                TIFIM.info           = entry.val;
            case 271
                TIFIM.make           = entry.val;
            case 273         % strip offset
                TIF.strip_offsets    = entry.val;
                TIF.num_strips       = entry.cnt;
                %disp(strcat('num_strips =', num2str(TIF.num_strips)));
            case 277         % sample_per pixel
                TIF.samples_per_pixel = entry.val;
                if (TIF.samples_per_pixel ~= 1)
                    error('color not supported');
                end
            case 278         % rows per strip
                TIF.rows_per_strip   = entry.val;
            case 279         % strip byte counts - number of bytes in each strip after any compressio
                TIF.strip_bytes      = entry.val;
            case 282        % X resolution
                TIFIM.x_resolution   = entry.val;
            case 283         % Y resolution
                TIFIM.y_resolution   = entry.val;
            case 296        % resolution unit
                TIFIM.resolution_unit= entry.val;
            case 305         % software
                TIFIM.software       = entry.val;
            case 306         % datetime
                TIFIM.datetime       = entry.val;
            case 315
                TIFIM.artist         = entry.val;
            case 317        %predictor for compression
                if (entry.val ~= 1)
                    error('unsupported predictor value');
                end
            case 320         % color map
                TIFIM.cmap          = entry.val;
                TIFIM.colors        = entry.cnt/3;
            case 339
                TIF.sample_format   = entry.val;
                if ( TIF.sample_format > 2 )
                    error('unsupported sample format = %i',TIF.sample_format);
                end
            case 33628       %metamorph specific data
                TIFIM.MM_private1   = entry.val;
            case 33629       %metamorph stack data?
                TIFIM.MM_stack      = entry.val;
                stack_cnt           = entry.cnt;
                %         disp([num2str(stack_cnt), ' frames, read:      ']);
            case 33630       %metamorph stack data: wavelength
                TIFIM.MM_wavelength = entry.val;
            case 33631       %metamorph stack data: gain/background?
                TIFIM.MM_private2   = entry.val;
            otherwise
                disp(sprintf('ignored tiff entry with tag %i cnt %i', TIF.entry_tag, entry.cnt));
        end
        % move to next IFD entry in the file
        fseek(TIF.file, file_pos+12,-1);
    end

    %read the next IFD address:
    ifd_pos = fread(TIF.file, 1, 'uint32', TIF.BOS);

    if img_last > stack_cnt
        img_last = stack_cnt;
    end

    stack_pos = 0;

    for i=1:stack_cnt

        if img_skip + 1 >= img_first
            img_read = img_read + 1;
            %disp(sprintf('reading MM frame %i at %i',num2str(img_read),num2str(TIF.strip_offsets(1)+stack_pos)));
            if (stack_cnt > 1)
                disp(sprintf('\b\b\b\b\b%4i', img_read));
            end
            TIFIM.data = read_strips(TIF, TIF.strip_offsets + stack_pos, TIFIM.width, TIFIM.height);
            S( img_read ) = TIFIM;

            %==============distribute the metamorph infos to each frame:
            if  isfield( TIFIM, 'MM_stack' )
                x = length(TIFIM.MM_stack) / stack_cnt;
                if  rem(x, 1) == 0
                    S( img_read ).MM_stack = TIFIM.MM_stack( 1+x*(img_read-1) : x*img_read );
                    if  isfield( TIFIM, 'info' )
                        x = length(TIFIM.info) / stack_cnt;
                        if rem(x, 1) == 0
                            S( img_read ).info = TIFIM.info( 1+x*(img_read-1) : x*img_read );
                        end
                    end
                end
            end
            if  isfield( TIFIM, 'MM_wavelength' )
                x = length(TIFIM.MM_wavelength) / stack_cnt;
                if rem(x, 1) == 0
                    S( img_read ).MM_wavelength = TIFIM.MM_wavelength( 1+x*(img_read-1) : x*img_read );
                end
            end

            if ( img_skip + img_read >= img_last )
                fclose(TIF.file);
                return;
            end
        else
            %disp('skiping strips');
            img_skip = img_skip + 1;
            skip_strips(TIF, TIF.strip_offsets + stack_pos);
        end
        stack_pos = ftell(TIF.file) - TIF.strip_offsets(1);
    end

end

fclose(TIF.file);

return;

function data = read_strips(TIF, strip_offsets, width, height)

% compute the width of each row in bytes:
numRows     = width * TIF.samples_per_pixel;
width_bytes = numRows * TIF.bytes_per_pixel;
numCols     = sum( TIF.strip_bytes / width_bytes ); %#ok Ignore MLint

typecode = sprintf('int%i', 8 * TIF.bytes_per_pixel / TIF.samples_per_pixel );
if TIF.sample_format == 1
    typecode = [ 'u', typecode ];
end

% Preallocate strip matrix:
data = eval( [ typecode, '(zeros(numRows, numCols));'] );

colIndx = 1;
for i = 1:TIF.num_strips
    fseek(TIF.file, strip_offsets(i), -1);
    strip = fread( TIF.file, TIF.strip_bytes(i) ./ TIF.bytes_per_pixel, typecode, TIF.BOS );
    if TIF.sample_format == 2
        %strip == bitcmp( strip );
    end

    if length(strip) ~= TIF.strip_bytes(i) / TIF.bytes_per_pixel
        error('End of file reached unexpectedly.');
    end
    stripCols = TIF.strip_bytes(i) ./ width_bytes;
    data(:, colIndx:(colIndx+stripCols-1)) = reshape(strip, numRows, stripCols);
    colIndx = colIndx + stripCols;
end
% Extract valid part of data
if ~all(size(data) == [width height]),
    data = data(1:width, 1:height);
    disp('extracting data');
end
% transpose the image
data = data';
return;


function skip_strips(TIF, strip_offsets)
fseek(TIF.file, strip_offsets(TIF.num_strips) + TIF.strip_bytes(TIF.num_strips),-1);
return;

%===================sub-functions that read an IFD entry:===================

function [nbbytes, typechar] = matlabtype(tifftypecode)
switch (tifftypecode)
    case 1
        nbbytes=1;
        typechar='uint8';
    case 2
        nbbytes=1;
        typechar='uchar';
    case 3
        nbbytes=2;
        typechar='uint16';
    case 4
        nbbytes=4;
        typechar='uint32';
    case 5
        nbbytes=8;
        typechar='uint32';
    otherwise
        error('tiff type not supported')
end
return


function  entry = readIFDentry(TIF)

entry.typecode = fread(TIF.file, 1, 'uint16', TIF.BOS);
entry.cnt      = fread(TIF.file, 1, 'uint32', TIF.BOS);
%disp(strcat('typecode =', num2str(entry.typecode),', cnt = ',num2str(entry.cnt)));
[ entry.nbbytes, entry.typechar ] = matlabtype(entry.typecode);
if entry.nbbytes * entry.cnt > 4
    % next field contains an offset:
    offset = fread(TIF.file, 1, 'uint32', TIF.BOS);
    %disp(strcat('offset = ', num2str(offset)));
    fseek(TIF.file, offset, -1);
end

if TIF.entry_tag == 33629   %special metamorph 'rationals'
    entry.val = fread(TIF.file, 6*entry.cnt, entry.typechar, TIF.BOS);
else
    if entry.typecode == 5
        entry.val = fread(TIF.file, 2*entry.cnt, entry.typechar, TIF.BOS);
    else
        entry.val = fread(TIF.file, entry.cnt, entry.typechar, TIF.BOS);
    end
end
if (entry.typecode == 2)
    entry.val = char(entry.val');
end

return;