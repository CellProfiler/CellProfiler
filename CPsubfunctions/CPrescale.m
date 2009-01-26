function [handles,OutputImage] = CPrescale(handles,InputImage,RescaleOption,MethodSpecificArguments)

% See the help for RESCALEINTENSITY for details.
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

if strncmpi(RescaleOption,'N',1) == 1
    OutputImage = InputImage;
elseif strncmpi(RescaleOption,'S',1) == 1
    %%% The minimum of the image is brought to zero, whether it
    %%% was originally positive or negative.
    IntermediateImage = InputImage - min(InputImage(:));
    %%% The maximum of the image is brought to 1.
    OutputImage = IntermediateImage ./ max(max(IntermediateImage));
elseif strncmpi(RescaleOption,'M',1) == 1
    %%% Rescales the image so the max equals the max of
    %%% the original image.
    if any(InputImage(:))
        IntermediateImage = InputImage ./ max(max(InputImage));
    else
        %% Image all zeros.  Leave it alone.
        IntermediateImage = InputImage;
    end
    OutputImage = IntermediateImage .* max(max(MethodSpecificArguments));
elseif strncmpi(RescaleOption,'G',1) == 1
    %%% Rescales the image so that all pixels are equal to or greater
    %%% than one. This is done by dividing each pixel of the image by
    %%% a scalar: the minimum pixel value anywhere in the smoothed
    %%% image. (If the minimum value is zero, .0001 is substituted
    %%% instead.) This rescales the image from 1 to some number. This
    %%% is useful in cases where other images will be divided by this
    %%% image, because it ensures that the final, divided image will
    %%% be in a reasonable range, from zero to 1.
    drawnow
    OutputImage = InputImage ./ max([min(min(InputImage)); .0001]);
    OutputImage(OutputImage<1) = 1;
elseif strncmpi(RescaleOption,'E',1) == 1
    LowestPixelOrig = MethodSpecificArguments{1};
    HighestPixelOrig = MethodSpecificArguments{2};
    LowestPixelRescale = MethodSpecificArguments{3};
    LowestPixelOrigPinnedValue = MethodSpecificArguments{4};
    HighestPixelRescale = MethodSpecificArguments{5};
    HighestPixelOrigPinnedValue = MethodSpecificArguments{6};
    ImageName = MethodSpecificArguments{5};
    % Case 1: Either of the arguments are AA
    if any([strcmp(upper(LowestPixelOrig), 'AA') strcmp(upper(HighestPixelOrig), 'AA')]),
        FindLowestIntensity =   strcmp(upper(LowestPixelOrig), 'AA');
        FindHighestIntensity =  strcmp(upper(HighestPixelOrig),'AA');
        
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            try
                %%% Notifies the user that the first image set will take much longer than
                %%% subsequent sets.
                %%% Obtains the screen size.
                [ScreenWidth,ScreenHeight] = CPscreensize;
                PotentialBottom = [0, (ScreenHeight-720)];
                BottomOfMsgBox = max(PotentialBottom);
                PositionMsgBox = [500 BottomOfMsgBox 350 100];
                h = CPmsgbox('Preliminary calculations are under way for the Rescale Intensity module.  Subsequent image sets will be processed much more quickly than the first image set.');
                set(h, 'Position', PositionMsgBox)
                drawnow
                %%% Retrieves the path where the images are stored from the handles
                %%% structure.
                fieldname = ['Pathname', ImageName];
                try Pathname = handles.Pipeline.(fieldname);
                catch error('Image processing was canceled because the Rescale Intensity module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Rescale Intensity module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Rescale Intensity module onward.')
                end
                %%% Retrieves the list of filenames where the images are stored from the
                %%% handles structure.
                fieldname = ['FileList', ImageName];
                FileList = handles.Pipeline.(fieldname);
                %%% Calculates the maximum and minimum pixel values based on all of the images.
                if (length(FileList) <= 0)
                    error('Image processing was canceled because the Rescale Intensity module found no images to process.');
                end
                maxPixelValue = -inf;
                minPixelValue = inf;
                for i=1:length(FileList)
                    Image = CPimread(fullfile(Pathname,char(FileList(i))));
                    if(max(max(Image)) > maxPixelValue)
                        maxPixelValue = max(max(Image));
                    end
                    if(min(min(Image)) < minPixelValue)
                        minPixelValue = min(min(Image));
                    end
                    drawnow
                end
            catch [ErrorMessage, ErrorMessage2] = lasterr;
                error(['An error occurred in the Rescale Intensity module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
            end
            HighestPixelOrig = double(maxPixelValue);
            LowestPixelOrig = double(minPixelValue);
            if FindHighestIntensity,
                fieldname = ['MaxPixelValue', ImageName];
                handles.Pipeline.(fieldname) = HighestPixelOrig;
            end
            if FindLowestIntensity,
                fieldname = ['MinPixelValue', ImageName];
                handles.Pipeline.(fieldname) = LowestPixelOrig;
            end
        else
            if FindHighestIntensity,
                fieldname = ['MaxPixelValue', ImageName];
                HighestPixelOrig = handles.Pipeline.(fieldname);
            end
            if FindLowestIntensity,
                fieldname = ['MinPixelValue',ImageName];
                LowestPixelOrig = handles.Pipeline.(fieldname);
            end
        end
    end
    
    % Case 2: Either of the arguments are AE
    if strcmp(upper(LowestPixelOrig), 'AE'),
        LowestPixelOrig = min(min(InputImage));
    end
    if strcmp(upper(HighestPixelOrig), 'AE'),
        HighestPixelOrig = max(max(InputImage));
    end
    
    % Case 3: Either of the arguments are numbers
    if isfinite(str2double(LowestPixelOrig))    % Evaulates to NaN if a string
        LowestPixelOrig = str2double(LowestPixelOrig);
    end
    if isfinite(str2double(HighestPixelOrig))
        HighestPixelOrig = str2double(HighestPixelOrig);
    end

    % Perform the rescaling
    InputImageMod = InputImage;
    % (1) Scale and shift the original image to produce the rescaled image.
    % Here, we find the linear transformation that maps the user-specified
    %   old high/low values to their new high/low values
    hi = HighestPixelOrig; HI = HighestPixelRescale;
    lo = LowestPixelOrig; LO = LowestPixelRescale;
    X = [lo 1; hi 1]\[LO; HI]; 
    OutputImage = InputImageMod*X(1) + X(2);
    % Extra measure to make sure values close to EPS are mapped to 0
    % (since the matrix algebra is not perfect)
    OutputImage(abs(OutputImage) > 0 & abs(OutputImage) < eps) = 0;
    
    % (2) Pixels above/below rescaled values are set to the desired pinning values 
    OutputImage(OutputImage > HighestPixelRescale) =   HighestPixelOrigPinnedValue;
    OutputImage(OutputImage < LowestPixelRescale) =    LowestPixelOrigPinnedValue;
    
elseif strncmpi(RescaleOption,'C',1) == 1
    OutputImage = uint8(InputImage*255);
elseif strncmpi(RescaleOption, 'T', 1) == 1
    TextName = MethodSpecificArguments;
    rescale = str2num(handles.Measurements.Image.(['LoadedText_',TextName]){handles.Current.SetBeingAnalyzed});
    OutputImage = InputImage / rescale;
else error(['For the rescaling option, you must enter N, S, M, G, E, C, or T for the method by which to rescale the image. Your entry was ', RescaleOption])
end