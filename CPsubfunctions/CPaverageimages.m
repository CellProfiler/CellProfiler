function [handles, OutputImage, ReadyFlag, MaskImage] = CPaverageimages(handles, Mode, ImageName, ProjectionImageName, MaskCountImageName)

% Note: Mode can be Accumulate or DoNow.
% ProjectionImageName is the name of the field to create in handles.Pipeline for the accumulated image
% MaskCountImageName is the name of the field to create in handles.Pipeline for the sum of the image masks
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

ReadyFlag = 'NotReady';

if strcmpi(Mode,'DoNow') == 1
    %%% Retrieves the path where the images are stored from the
    %%% handles structure.
    fieldname = ['Pathname', ImageName];
    try Pathname = handles.Pipeline.(fieldname);
    catch error('Image processing was canceled because the CPaverageimages subfunction (which is used by Make Projection and Correct Illumination modules) uses all the images in a set in its calculations. Therefore, the entire image set to be averaged must exist prior to processing the first image set through the pipeline. In other words, the module using CPaverageimages must be run straight from a LoadImages module rather than following an image analysis module. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this module onward.')
    end
    %%% Retrieves the list of filenames where the images are stored
    %%% from the handles structure.
    fieldname = ['FileList', ImageName];
    try FileList = handles.Pipeline.(fieldname);
    catch
        error(['Image processing was canceled because the CPaverageimages subfunction (which is used by Make Projection and Correct Illumination modules) could not find the input image. CellProfiler expected to find an image named "', ImageName, '" but that image has not been created by the pipeline. Please adjust your pipeline to produce the image "', ImageName, '" prior to the use of the CPaverageimages subfunction.'])
    end
    %%% Calculates the mean image. Initializes the variable.
    TotalImage = CPimread(fullfile(Pathname,char(FileList(1))));
    %%% Waitbar shows the percentage of image sets remaining.
    WaitbarHandle = waitbar(0,'');
    %%% Obtains the screen size and determines where the wait bar will
    %%% be displayed.
    [ScreenWidth,ScreenHeight] = CPscreensize;
    PotentialBottom = [0, (ScreenHeight-720)];
    BottomOfMsgBox = max(PotentialBottom);
    OldPos = get(WaitbarHandle,'position');
    PositionMsgBox = [250 BottomOfMsgBox OldPos(3) 90];
    set(WaitbarHandle,'color',[.7 .7 .9]);
    userData.Application = 'CellProfiler';
    set(WaitbarHandle,'UserData',userData);
    set(WaitbarHandle, 'Position', PositionMsgBox)
    drawnow
    TimeStart = clock;
    NumberOfImages = length(FileList);
    for i=2:length(FileList)
        OrigImage = fullfile(Pathname,char(FileList(i)));
        %%% Checks that the original image is two-dimensional (i.e.
        %%% not a color image), which would disrupt several of the
        %%% image functions.
        if ndims(OrigImage) ~= 2
            error('Image processing was canceled because calculating the average image (which is used by the Average and Correct Illumination modules) requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
        TotalImage = TotalImage + CPimread(OrigImage);
        CurrentTime = clock;
        TimeSoFar = etime(CurrentTime,TimeStart);
        TimePerSet = TimeSoFar/i;
        ImagesRemaining = NumberOfImages - i;
        TimeRemaining = round(TimePerSet*ImagesRemaining);
        WaitbarText = {'Calculating the average image.'; 'Subsequent image sets will be processed';'more quickly than the first image set.'; ['Seconds remaining: ', num2str(TimeRemaining),]};
        WaitbarText = char(WaitbarText);
        waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
        drawnow
    end
    if length(FileList) == 1
        CurrentTime = clock;
        TimeSoFar = etime(CurrentTime,TimeStart);
    end
    WaitbarText = {'Calculations of the average image are finished.'; 'Subsequent image sets will be processed';'more quickly than the first image set.';['Seconds consumed: ',num2str(TimeSoFar),]};
    WaitbarText = char(WaitbarText);
    waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
    OutputImage = TotalImage / length(FileList);
    MaskImage = ones(size(OutputImage));
    ReadyFlag = 'Ready';

elseif strcmpi(Mode,'Accumulate') == 1
    %%% In Pipeline (cycling) mode, each time through the image sets,
    %%% the image is added to the existing cumulative image. Reads
    %%% (opens) the image you want to analyze and assigns it to a
    %%% variable.
    fieldname = ['', ImageName];
    mask_fieldname = ['CropMask', ImageName];
    has_mask = isfield(handles.Pipeline,mask_fieldname);
    %%% Performs certain error-checking and initializing functions the
    %%% first time throught the image set.
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Checks whether the image to be analyzed exists in the
        %%% handles structure.
        if isfield(handles.Pipeline, ImageName)==0,
            %%% If the image is not there, an error message is
            %%% produced.  The error is not displayed: The error
            %%% function halts the current function and returns
            %%% control to the calling function (the analyze all
            %%% images button callback.)  That callback recognizes
            %%% that an error was produced because of its try/catch
            %%% loop and breaks out of the image analysis loop without
            %%% attempting further modules.
            error(['Image processing was canceled because the average function (which is used by the Average and Correct Illumination modules) could not find the input image.  CellProfiler expected to find an image named "', ImageName, '" but that image has not been created by the pipeline. Please adjust your pipeline to produce the image "', ImageName, '" prior to the use of the average function.'])
        end
        %%% Retrieves the current image.
        OrigImage = handles.Pipeline.(fieldname);
        %%% Creates the empty variable so it can be retrieved later
        %%% without causing an error on the first image set.
        handles.Pipeline.(ProjectionImageName) = zeros(size(OrigImage));
        handles.Pipeline.(MaskCountImageName) = zeros(size(OrigImage));
    end
    %%% Retrieves the current image.
    OrigImage = handles.Pipeline.(fieldname);
    %%% Checks that the original image is two-dimensional (i.e. not a
    %%% color image), which would disrupt several of the image
    %%% functions.
    if ndims(OrigImage) ~= 2
        error('Image processing was canceled because the average function (which is used by the Average and Correct Illumination modules) requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
    end
    %%% Retrieves the existing projection image, as accumulated so
    %%% far.
    ProjectedImage = handles.Pipeline.(ProjectionImageName);
    if has_mask
        mask = handles.Pipeline.(mask_fieldname);
        OutputImage = ProjectedImage + OrigImage .* mask;
        handles.Pipeline.(MaskCountImageName) = handles.Pipeline.(MaskCountImageName)+mask;
        MaskImage = (handles.Pipeline.(MaskCountImageName) > 0);
        OutputImage = OutputImage./max(handles.Pipeline.(MaskCountImageName),1);
        if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
            %%% Divides by the total number of images in order to average.
            ReadyFlag = 'Ready';
        end
    else
        %%% Adds the current image to it.
        OutputImage = ProjectedImage + OrigImage;

        %%% If the last image set has just been processed, indicate that
        %%% the projection image is ready.
        MaskImage = ones(size(OutputImage));
        if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
            %%% Divides by the total number of images in order to average.
            OutputImage = OutputImage/handles.Current.NumberOfImageSets;
            MaskImage = ones(size(OutputImage));
            ReadyFlag = 'Ready';
        end
    end
    %%% Saves the updated projection image to the handles structure.
    handles.Pipeline.(ProjectionImageName) = OutputImage;
end