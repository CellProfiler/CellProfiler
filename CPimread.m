function LoadedImage = CPimread(varargin)

if nargin == 2,
    CurrentFileName = varargin{1};
    handles = varargin{2};
    %%% Handles a non-Matlab readable file format.
    if isfield(handles.Pipeline, 'DIBwidth') == 1
        %%% Opens this non-Matlab readable file format.
        Width = handles.Pipeline.DIBwidth;
        Height = handles.Pipeline.DIBheight;
        Channels = handles.Pipeline.DIBchannels;
        BitDepth = handles.Pipeline.DIBbitdepth;
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
    else
        %%% Opens Matlab-readable file formats.
        try
            %%% Read (open) the image you want to analyze and assign it to a variable,
            %%% "LoadedImage".
            LoadedImage = im2double(imread(char(CurrentFileName)));
        catch error(['Image processing was canceled because the module could not load the image "', char(CurrentFileName), '" in directory "', pwd,'.  The error message was "', lasterr, '"'])
        end
    end
else
    CurrentFileName = varargin{1};
    [Pathname, FileName, ext] = fileparts(char(CurrentFileName));
    if strcmp('.DIB', upper(ext)),
        %%% Opens this non-Matlab readable file format.
        Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
        Width = str2double(Answers{1});
        Height = str2double(Answers{2});
        BitDepth = str2double(Answers{3});
        Channels = str2double(Answers{4});
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
        %%% Opens Matlab-readable file formats.
        try
            %%% Read (open) the image you want to analyze and assign it to a variable,
            %%% "LoadedImage".
            LoadedImage = im2double(imread(char(CurrentFileName)));
        catch error(['Unable to open the file, perhaps not a valid Image File.']);
        end
    end
end