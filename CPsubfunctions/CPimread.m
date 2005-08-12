function [LoadedImage, handles] = CPimread(varargin)

if nargin == 0 %returns the vaild image extensions
    formats = imformats;
    LoadedImage = [cat(2, formats.ext) {'dib'} {'mat'}]; %LoadedImage is not a image here, but rather a set
    return
elseif nargin == 2,
    CurrentFileName = varargin{1};
    handles = varargin{2};
    %%% Handles a non-Matlab readable file format.
    [Pathname, FileName, ext] = fileparts(char(CurrentFileName));
    if strcmp('.DIB', upper(ext)),
    %%% Opens this non-Matlab readable file format.
        try
            Width = handles.Pipeline.DIBwidth;
            Height = handles.Pipeline.DIBheight;
            Channels = handles.Pipeline.DIBchannels;
            BitDepth = handles.Pipeline.DIBbitdepth;
        catch
            Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
            try
                handles.Pipeline.DIBwidth = str2double(Answers{1});
                Width=handles.Pipeline.DIBwidth;
                handles.Pipeline.DIBheight = str2double(Answers{2});
                Height=handles.Pipeline.DIBheight;
                handles.Pipeline.DIBbitdepth = str2double(Answers{3});
                BitDepth=handles.Pipeline.DIBbitdepth;
                handles.Pipeline.DIBchannels = str2double(Answers{4});
                Channels=handles.Pipeline.DIBchannels;
            catch
                return %If the user cancels or closes the window.
            end
        end
        fid = fopen(char(CurrentFileName), 'r');
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
    elseif strcmp('.MAT',upper(ext))
        load(CurrentFileName);
        if exist('Image')
            LoadedImage = Image; 
        else
            error('Was unable to load the image.  This could be because the .mat file specified is not a proper image file');
        end
    else 
        try
            %%% Read (open) the image you want to analyze and assign it to a variable,
            %%% "LoadedImage".
            %%% Opens Matlab-readable file formats.
            LoadedImage = im2double(imread(char(CurrentFileName)));
        catch
            error(['Image processing was canceled because the module could not load the image "', char(CurrentFileName), '" in directory "', pwd,'.  The error message was "', lasterr, '"'])
        end
    end
else
    CurrentFileName = varargin{1};
    [Pathname, FileName, ext] = fileparts(char(CurrentFileName));
    if strcmp('.DIB', upper(ext)),
        %%% Opens this non-Matlab readable file format.
        Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
        try
            Width = str2double(Answers{1});
            Height = str2double(Answers{2});
            BitDepth = str2double(Answers{3});
            Channels = str2double(Answers{4});
        catch
            return %If the user cancels or closes the window.
        end
        fid = fopen(char(CurrentFileName), 'r');
        if (fid == -1),
            error(['The file ', FileName, ' could not be opened. CellProfiler attempted to open it in DIB file format.']);
        end
        fread(fid, 52, 'uchar');
        LoadedImage = zeros(Height,Width,Channels);
        for c=1:Channels,
            [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
            if Count < (Width * Height),
                fclose(fid);
                error(['End-of-file encountered while reading ', FileName, '. Have you entered the proper size and number of channels for these images?']);
            end
            LoadedImage(:,:,c) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
        end
        fclose(fid);
    else
        try
            %%% Read (open) the image you want to analyze and assign it to a variable,
            %%% "LoadedImage".
            %%% Opens Matlab-readable file formats.
            LoadedImage = im2double(imread(char(CurrentFileName)));
        catch
            error(['Image processing was canceled because the module could not load the image "', char(CurrentFileName), '" in directory "', pwd,'.  The error message was "', lasterr, '"'])
        end
    end
end