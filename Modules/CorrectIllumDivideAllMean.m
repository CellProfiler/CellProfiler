function handles = AlgCorrectIllumDivideAllMean(handles)

% Help for the Correct Illumination Divide All Mean module: 
% Category: Pre-processing
%
% This module corrects for uneven illumination of each image, based on
% information from a set of images collected at the same time. 
%
% How it works:
% This module works by averaging together all of the images, then
% smoothing this image by fitting a low-order polynomial to the
% resulting average image and rescaling it.  This produces an image
% that represents the variation in illumination across the field of
% view.  This process is carried out before the first image set is
% processed; subsequent image sets use the already calculated image.
% Each image is divided by this illumination image to produce the
% corrected image.
%
% If you want to run this module only to calculate the mean and
% illumination images and not to correct every image in the directory,
% simply run the module as usual and use the button on the Timer to
% stop processing after the first image set.
%
% SAVING IMAGES: The illumination corrected images produced by this
% module can be easily saved using the Save Images module, using the
% name you assign. The mean image can be saved using the name
% MeanImageAD plus whatever you called the corrected image (e.g.
% MeanImageADCorrBlue). The Illumination correction image can be saved
% using the name IllumImageAD plus whatever you called the corrected
% image (e.g. IllumImageADCorrBlue).  Note that using the Save Images
% module saves a copy of the image in an image file format, which has
% lost some of the detail that a matlab file format would contain.  In
% other words, if you want to save the illumination image to use it in
% a later analysis, you should use the settings boxes within this
% module to save the illumination image in '.mat' format. If you want
% to save other intermediate images, alter the code for this module to
% save those images to the handles structure (see the SaveImages
% module help) and then use the Save Images module.
%
% See also ALGCORRECTILLUMDIVIDEALLMEANRETRIEVEIMG,
% ALGCORRECTILLUMSUBTRACTALLMIN,
% ALGCORRECTILLUMDIVIDEEACHMIN_9, ALGCORRECTILLUMDIVIDEEACHMIN_10,
% ALGCORRECTILLUMSUBTRACTEACHMIN.

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

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of: purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.  Do not
% change the programming notes in any modules! These are standard
% across all modules for maintenance purposes, so anything
% module-specific should be kept separate.

% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
CorrectedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR08 = To save the illum. corr. image to use later, type a file name + .mat. Else, 'N'
%defaultVAR08 = N
IllumCorrectFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = If you have already created an illumination correction image to be used, enter the 
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
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Correct Illumination module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

        
%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING: 
% figure, imshow(BlurredImage, []), title('BlurredImage') 
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING: 
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

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
            fieldname = ['Pathname', ImageName];
            try Pathname = handles.Pipeline.(fieldname);
            catch error('Image processing was canceled because the Correct Illumination module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Correct Illumination module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Correct Illumination module onward.')
            end
            %%% Changes to that directory.
            cd(Pathname)
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['FileList', ImageName];
            FileList = handles.Pipeline.(fieldname);
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
        fieldname = ['MeanImageAD', CorrectedImageName];
        handles.Pipeline.(fieldname) = MeanImage;        
    end
    fieldname = ['IllumImageAD', CorrectedImageName];
    handles.Pipeline.(fieldname) = IlluminationImage;
end

%%% The following is run for every image set. Retrieves the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['MeanImageAD', CorrectedImageName];
if isfield(handles.Pipeline, fieldname) == 1
    MeanImage = handles.Pipeline.(fieldname);
end
fieldname = ['IllumImageAD', CorrectedImageName];
IlluminationImage = handles.Pipeline.(fieldname);
%%% Corrects the original image based on the IlluminationImage,
%%% by dividing each pixel by the value in the IlluminationImage.
CorrectedImage = OrigImage ./ IlluminationImage;

%%% Returns to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this algorithm. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
if any(findobj == ThisAlgFigureNumber) == 1;
% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisAlgFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
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

% PROGRAMMING NOTE
% HANDLES STRUCTURE:
%       In CellProfiler (and Matlab in general), each independent
% function (module) has its own workspace and is not able to 'see'
% variables produced by other modules. For data or images to be shared
% from one module to the next, they must be saved to what is called
% the 'handles structure'. This is a variable, whose class is
% 'structure', and whose name is handles. Data which should be saved
% to the handles structure within each module includes: any images,
% data or measurements which are to be eventually saved to the hard
% drive (either in an output file, or using the SaveImages module) or
% which are to be used by a later module in the analysis pipeline. Any
% module which produces or passes on an image needs to also pass along
% the original filename of the image, named after the new image name,
% so that if the SaveImages module attempts to save the resulting
% image, it can be named by appending text to the original file name.
%       It is important to think about which of these data should be
% deleted at the end of an analysis run because of the way Matlab
% saves variables: For example, a user might process 12 image sets of
% nuclei which results in a set of 12 measurements ("TotalNucArea")
% stored in the handles structure. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "TotalNucArea" to
% analyze 4 image sets, the 4 measurements will overwrite the first 4
% measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only an algorithm which
% depends on the output "SegmNucImg" but does not run an algorithm
% that produces an image by that name, the algorithm will run just
% fine: it will just repeatedly use the processed image of nuclei
% leftover from the last image set, which was left in the handles
% structure ("SegmNucImg").
%
% INCLUDE FURTHER DESCRIPTION OF MEASUREMENTS PER CELL AND PER IMAGE
% HERE>>>
%
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, it is better to store a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%
%       Extracting measurements: handles.dMCCenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.dMCAreaNuclei{2}(1) gives the area of the first object in
% the second image.

%%% Saves the corrected image to the
%%% handles structure so it can be used by subsequent algorithms.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the corrected image name.
fieldname = ['Filename', CorrectedImageName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;