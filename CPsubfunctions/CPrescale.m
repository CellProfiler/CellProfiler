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
    IntermediateImage = InputImage ./ max(max(InputImage));
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
    HighestPixelRescale = MethodSpecificArguments{4};
    ImageName = MethodSpecificArguments{5};
    if (strcmp(upper(LowestPixelOrig), 'AA') & strcmp(upper(HighestPixelOrig), 'AA')) == 1
        if handles.Current.SetBeingAnalyzed == 1
            try
                %%% Notifies the user that the first image set will take much longer than
                %%% subsequent sets.
                %%% Obtains the screen size.
                ScreenSize = get(0,'ScreenSize');
                ScreenHeight = ScreenSize(4);
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
            fieldname = ['MaxPixelValue', ImageName];
            handles.Pipeline.(fieldname) = HighestPixelOrig;
            fieldname = ['MinPixelValue', ImageName];
            handles.Pipeline.(fieldname) = LowestPixelOrig;
        else
            fieldname = ['MaxPixelValue', ImageName];
            HighestPixelOrig = handles.Pipeline.(fieldname);
            fieldname = ['MinPixelValue',ImageName];
            LowestPixelOrig = handles.Pipeline.(fieldname);
        end
    elseif (strcmp(upper(LowestPixelOrig), 'AE') & strcmp(upper(HighestPixelOrig), 'AE'))== 1
        LowestPixelOrig = min(min(MethodSpecificArguments));
        HighestPixelOrig = max(max(MethodSpecificArguments));
    else
        LowestPixelOrig = str2double(LowestPixelOrig);
        HighestPixelOrig = str2double(HighestPixelOrig);
    end
    %%% Rescales the Image.
    InputImageMod = InputImage;
    %Any pixel in the original image lower than the user-input lowest bound is
    %pinned to the lowest value.
    InputImageMod(InputImageMod < LowestPixelOrig) = LowestPixelOrig;
    %Any pixel in the original image higher than the user-input highest bound is
    %pinned to the lowest value.
    InputImageMod(InputImageMod > HighestPixelOrig) = HighestPixelOrig;
    %Scales and shifts the original image to produce the rescaled image
    scaleFactor = (HighestPixelRescale - LowestPixelRescale)  / (HighestPixelOrig - LowestPixelOrig);
    shiftFactor = LowestPixelRescale - LowestPixelOrig;
    OutputImage = InputImageMod + shiftFactor;
    OutputImage = OutputImage * scaleFactor;
elseif strncmpi(RescaleOption,'C',1) == 1
    OutputImage = uint8(InputImage*255);
else error(['For the rescaling option, you must enter N, S, M, G, E, or C for the method by which to rescale the image. Your entry was ', RescaleOption])
end