function handles = AlgCorrectIllumDivideAllMean(handles)

% Help for the Correct Illumination Divide All Mean module: 
%
% This module corrects for uneven illumination of each image, based on
% information from a set of images collected at the same time. This module
% works by averaging together all of the images, then smoothing this image
% by fitting a low-order polynomial to the resulting average image and
% rescaling it.  This produces an image that represents the variation in
% illumination across the field of view.  This process is carried out
% before the first image set is processed; subsequent image sets use the
% already calculated image. Each image is divided by this illumination
% image to produce the corrected image.
%
% If you want to run this module only to calculate the mean and
% illumination images and not to correct every image in the directory,
% simply run the module as usual and use the button on the Timer to
% stop processing after the first image set.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Correct Illumination Divide All Mean module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
CorrectedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR08 = To save the illum. corr. image to use later, type a file name + .mat. Else, 'N'
%defaultVAR08 = N
IllumCorrectFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = If you have already created an illumination corrrection image to be used, enter the 
%textVAR10 = path & file name of the image below. To calculate the illumination correction image 
%textVAR11 = from all the images of this color that will be processed, leave a slash in the box below.
%defaultVAR11 = /
IllumCorrectPathAndFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Correct Illumination module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.(fieldname);
        
%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
%%% Make note of the current directory so the module can return to it
%%% at the end of this module.
CurrentDirectory = cd;

%%% The first time the module is run, the image to be used for
%%% correction must be retrieved from a file or calculated.
if handles.setbeinganalyzed == 1
    %%% If the user has specified a path and file name of an illumination
    %%% correction image that has already been created, the image is
    %%% loaded.
    if strcmp(IllumCorrectPathAndFileName, '/') ~= 1
        try StructureIlluminationImage = load(IllumCorrectPathAndFileName);
             IlluminationImage = StructureIlluminationImage.IlluminationImage;
        catch error(['Image processing was canceled because there was a problem loading the image ', IllumCorrectPathAndFileName, '. Check that the full path and file name has been typed correctly.'])
        end
        %%% Otherwise, the illumination correction image is calculated based on all
        %%% the images of this type that will be processed.
    else 
        try
            %%% Obtains the screen size and determines where the wait bar
            %%% will be displayed.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            %%% Retrieves the path where the images are stored from the handles
            %%% structure.
            fieldname = ['dOTPathName', ImageName];
            try PathName = handles.(fieldname);
            catch error('Image processing was canceled because the Correct Illumination module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Correct Illumination module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Correct Illumination module onward.')
            end
            %%% Changes to that directory.
            cd(PathName)
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['dOTFileList', ImageName];
            FileList = handles.(fieldname);
            %%% Calculates the mean image.  If cells are distributed uniformly in
            %%% the images, the mean of all the images should be a good
            %%% estimate of the illumination.
            TotalImage = im2double(imread(char(FileList(1))));
            %%% Waitbar shows the percentage of image sets remaining.
            WaitbarHandle = waitbar(0,'');
            set(WaitbarHandle, 'Position', PositionMsgBox)
            drawnow
            TimeStart = clock;
            NumberOfImages = length(FileList);
            for i=2:length(FileList)
                TotalImage = TotalImage + im2double(imread(char(FileList(i))));
                CurrentTime = clock;
                TimeSoFar = etime(CurrentTime,TimeStart);
                TimePerSet = TimeSoFar/i;
                ImagesRemaining = NumberOfImages - i;
                TimeRemaining = round(TimePerSet*ImagesRemaining);
                WaitbarText = {'Calculating the illumination function for the';'Correct Illumination All Divide module.'; 'Subsequent image sets will be processed';'more quickly than the first image set.'; ['Seconds remaining: ', num2str(TimeRemaining),]};
                WaitbarText = char(WaitbarText);
                waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
                drawnow
            end
            if length(FileList) == 1
                CurrentTime = clock;
                TimeSoFar = etime(CurrentTime,TimeStart);
            end
            WaitbarText = {'Calculations of the illumination function are finished for the';'Correct Illumination All Divide module.'; 'Subsequent image sets will be processed';'more quickly than the first image set.';['Seconds consumed: ',num2str(TimeSoFar),]};
            WaitbarText = char(WaitbarText);
            waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
            MeanImage = TotalImage / length(FileList);
            %%% The following is used to fit a low-dimensional polynomial to the mean image.
            %%% The result, IlluminationImage, is an image of the smooth illumination function.
            [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
            x2 = x.*x;
            y2 = y.*y;
            xy = x.*y;
            o = ones(size(MeanImage));
            Ind = find(MeanImage > 0);
            Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(MeanImage(Ind));
            drawnow
            IlluminationImage1 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(MeanImage));
            %%% The final IlluminationImage is produced by dividing each
            %%% pixel of the illumination image by a scalar: the minimum
            %%% pixel value anywhere in the illumination image. (If the
            %%% minimum value is zero, .00000001 is substituted instead.)
            %%% This rescales the IlluminationImage from 1 to some number.
            %%% This ensures that the final, corrected image will be in a
            %%% reasonable range, from zero to 1.
            drawnow
            IlluminationImage = IlluminationImage1 ./ max([min(min(IlluminationImage1)); .00000001]);
               %%% Note: the following "imwrite" saves the illumination
            %%% correction image in TIF format, but the image is compressed
            %%% so it is not as smooth as the image that is saved using the
            %%% "save" function below, which is stored in matlab ".mat"
            %%% format.
            % imwrite(IlluminationImage, 'IlluminationImage.tif', 'tif')
            
            %%% Saves the illumination correction image to the hard
            %%% drive if requested.
            if strcmp(upper(IllumCorrectFileName), 'N') == 0
                try
                    save(IllumCorrectFileName, 'IlluminationImage')
                catch error(['There was a problem saving the illumination correction image to the hard drive. The attempted filename was ', IllumCorrectFileName, '.'])
                end
            end
        catch [ErrorMessage, ErrorMessage2] = lasterr;
            error(['An error occurred in the Correct Illumination module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
        end
    end    
    %%% Stores the mean image and the Illumination image to the handles
    %%% structure.
    if exist('MeanImage','var') == 1
        fieldname = ['dOTMeanImageAD', ImageName];
        handles.(fieldname) = MeanImage;        
    end
    fieldname = ['dOTIllumImageAD', ImageName];
    handles.(fieldname) = IlluminationImage;
end

%%% The following is run for every image set. Retrieves the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['dOTMeanImageAD', ImageName];
if isfield(handles, fieldname) == 1
    MeanImage = handles.(fieldname);
end
fieldname = ['dOTIllumImageAD', ImageName];
IlluminationImage = handles.(fieldname);
%%% Corrects the original image based on the IlluminationImage,
%%% by dividing each pixel by the value in the IlluminationImage.
CorrectedImage = OrigImage ./ IlluminationImage;

%%% Returns to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% The mean image does not absolutely have to be present in order to
    %%% carry out the calculations if the illumination image is provided,
    %%% so the following subplot is only shown if MeanImage exists in the
    %%% workspace.
    subplot(2,2,2); imagesc(CorrectedImage); 
    title('Illumination Corrected Image');
    if exist('MeanImage','var') == 1
        subplot(2,2,3); imagesc(MeanImage); 
        title(['Mean of all ', ImageName, ' images']);
    end
    subplot(2,2,4); imagesc(IlluminationImage); 
    title('Illumination Function'); colormap(gray)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', CorrectedImageName];
handles.(fieldname) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;