function handles = AlgCorrectIlluminationDivideAllMean10(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR1 = What did you call the image to be corrected?
%defaultVAR1 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR2 = What do you want to call the corrected image?
%defaultVAR2 = CorrBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
CorrectedImageName = handles.(fieldname);
%textVAR4 = To save each corrected image, enter text to append to the image name 
%defaultVAR4 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
SaveImage = handles.(fieldname);
%textVAR5 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR6 = In what file format do you want to save images? Do not include a period
%defaultVAR6 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
FileFormat = handles.(fieldname);
%textVAR8 = To save the illum. corr. image to use later, type a file name + .mat. Else, 'N'
%defaultVAR8 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
IllumCorrectFileName = handles.(fieldname);
%textVAR9 = If you have already created an illumination corrrection image to be used, enter the 
%textVAR10 = path & file name of the image below. To calculate the illumination correction image 
%textVAR11 = from all the images of this color that will be processed, leave a slash in the box below.
%defaultVAR11 = /
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
IllumCorrectPathAndFileName = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Correct Illumination module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Correct Illumination module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Read the image.
OrigImage = handles.(fieldname);
        % figure, imshow(OrigImage), title('OrigImage')
        
%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Find and remove the file format extension within the original file
%%% name, but only if it is at the end. Strip the original file format extension 
%%% off of the file name, if it is present, otherwise, leave the original
%%% name intact.
CharFileName = char(FileName);
PotentialDot = CharFileName(end-3:end-3);
if strcmp(PotentialDot,'.') == 1
    BareFileName = CharFileName(1:end-4);
else BareFileName = CharFileName;
end
%%% Assemble the new image name.
NewImageName = [BareFileName,SaveImage,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveImage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Correct Illumination All Divide module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Correct Illumination All Divide module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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
            % figure, imagesc(IlluminationImage), colormap('gray'), title('Loaded Illumination Correction Image')
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
            set(WaitbarHandle, 'Position', [PositionMsgBox])
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
            end
            if length(FileList) == 1
                CurrentTime = clock;
                TimeSoFar = etime(CurrentTime,TimeStart);
            end
            WaitbarText = {'Calculations of the illumination function are finished for the';'Correct Illumination All Divide module.'; 'Subsequent image sets will be processed';'more quickly than the first image set.';['Seconds consumed: ',num2str(TimeSoFar),]};
            WaitbarText = char(WaitbarText);
            waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
            MeanImage = TotalImage / length(FileList);
            %    figure, imagesc(MeanImage), colormap(gray)
            % imwrite(MeanImage/256, [ImageName,'MeanImage2','.',FileFormat], FileFormat);
            %%% The following is used to fit a low-dimensional polynomial to the mean image.
            %%% The result, IlluminationImage, is an image of the smooth illumination function.
            [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
            x2 = x.*x;
            y2 = y.*y;
            xy = x.*y;
            o = ones(size(MeanImage));
            Ind = find(MeanImage > 0);
            Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(MeanImage(Ind));
            IlluminationImage1 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(MeanImage));
               % figure, imagesc(IlluminationImage), colormap(gray), title('Calculated illumination correction image')
            %%% The final IlluminationImage is produced by dividing each
            %%% pixel of the illumination image by a scalar: the minimum
            %%% pixel value anywhere in the illumination image. (If the
            %%% minimum value is zero, .00000001 is substituted instead.)
            %%% This rescales the IlluminationImage from 1 to some number.
            %%% This ensures that the final, corrected image will be in a
            %%% reasonable range, from zero to 1.
            IlluminationImage = IlluminationImage1 ./ max([min(min(IlluminationImage1)); .00000001]);
               %%% Note: the following "imwrite" saves the illumination
            %%% correction image in TIF format, but the image is compressed
            %%% so it is not as smooth as the image that is saved using the
            %%% "save" function below, which is stored in matlab ".mat"
            %%% format.
            % imwrite(IlluminationImage, 'IlluminationImage.tif', 'tif')
            
            %%% The illumination correction image is saved to the hard
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
    %%% Store the mean image and the Illumination image to the handles
    %%% structure. I have given it a specific name to ensure that the
    %%% user hasn't chosen this exact name for something else.
    if exist('MeanImage') == 1
        fieldname = ['dOTMeanImageAD', ImageName];
        handles.(fieldname) = MeanImage;        
    end
    fieldname = ['dOTIllumImageAD', ImageName];
    handles.(fieldname) = IlluminationImage;
    %%% Update the handles structure.
    %%% Removed for parallel: guidata(gcbo, handles);
end

%%% The following is run for every image set. Retrieve the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['dOTMeanImageAD', ImageName];
if isfield(handles, fieldname) == 1
    MeanImage = handles.(fieldname);
end
fieldname = ['dOTIllumImageAD', ImageName];
IlluminationImage = handles.(fieldname);
%%% The original image is corrected based on the IlluminationImage,
%%% by dividing each pixel by the value in the IlluminationImage.
CorrectedImage = OrigImage ./ IlluminationImage;

%%% Checking to see whether the rescaling makes sense:
% MAX(1) = max(max(OrigImage));
% MIN(1) = min(min(OrigImage));
% MAX(2) = max(max(IlluminationImage));
% MIN(2) = min(min(IlluminationImage));
% MAX(3) = max(max(CorrectedImage))
% MIN(3) = min(min(CorrectedImage))

%%% Returns to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed
%%% the figure window, so do not do any important calculations here.
%%% Otherwise an error message will be produced if the user has closed the
%%% window but you have attempted to access data that was supposed to be
%%% produced by this part of the code.

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
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
    if exist('MeanImage') == 1
        subplot(2,2,3); imagesc(MeanImage); 
        title(['Mean of all ', ImageName, ' images']);
    end
    subplot(2,2,4); imagesc(IlluminationImage); 
    title('Illumination Function'); colormap(gray)
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The corrected image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', CorrectedImageName];
handles.(fieldname) = CorrectedImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% The original file name is saved to the handles structure in a
%%% field named after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the corrected image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(CorrectedImage, NewImageName, FileFormat);
min(min(CorrectedImage))
max(max(CorrectedImage))
min(min(uint8(CorrectedImage)))
max(max(uint8(CorrectedImage)))

end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Correct Illumination Divide All Mean module: 

%%%%%   This module corrects for uneven illumination of each image, based on
%%%%% information from a set of images collected at the same time.
%%%%%   This module works by averaging together all of the images, then
%%%%% smoothing this image by fitting a low-order polynomial to the
%%%%% resulting average image and rescaling it.  This produces an image
%%%%% that represents the variation in illumination across the field of
%%%%% view.  This process is carried out before the first image set is
%%%%% processed; subsequent image sets use the already calculated image.
%%%%% Each image is divided by this illumination image to produce the
%%%%% corrected image.
%%%%% .
%%%%% If you want to run this module only to calculate the mean and
%%%%% illumination images and not to correct every image in the directory,
%%%%% simply run the module as usual and use the button on the Timer to
%%%%% stop processing after the first image set.
%%%%% . 
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified at the top of the
%%%%% CellProfiler window.
%%%%% .
%%%%% If you want to save other processed images, open the m-file for this 
%%%%% image analysis module, go to the line in the
%%%%% m-file where the image is generated, and there should be 2 lines
%%%%% which have been inactivated.  These are green comment lines that are
%%%%% indented. To display an image, remove the percent sign before
%%%%% the line that says "figure, imshow...". This will cause the image to
%%%%% appear in a fresh display window for every image set. To save an
%%%%% image to the hard drive, remove the percent sign before the line
%%%%% that says "imwrite..." and adjust the file type and appendage to the
%%%%% file name as desired.  When you have finished removing the percent
%%%%% signs, go to File > Save As and save the m file with a new name.
%%%%% Then load the new image analysis module into the CellProfiler as
%%%%% usual.
%%%%% Please note that not all of these imwrite lines have been checked for
%%%%% functionality: it may be that you will have to alter the format of
%%%%% the image before saving.  Try, for example, adding the uint8 command:
%%%%% uint8(Image) surrounding the image prior to using the imwrite command
%%%%% if the image is not saved correctly.