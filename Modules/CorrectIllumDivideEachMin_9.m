function handles = AlgCorrectIlluminationDivideEachMin_9(handles)

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

%textVAR4 = To save the corrected image, enter text to append to the image name 
%defaultVAR4 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
SaveImage = handles.(fieldname);
%textVAR5 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR6 = In what file format do you want to save images? Do not include a period
%defaultVAR6 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
FileFormat = handles.(fieldname);

%textVAR8 = Block size. This should be set large enough that every square block 
%defaultVAR8 = 60
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
BlockSize = str2num(handles.(fieldname));
%textVAR9 = of pixels is likely to contain some background.

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
        
%%% Check whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
MinLengthWidth = min(m,n);
if BlockSize >= MinLengthWidth
        error('Image processing was canceled because in the Correct Illumination module the selected block size is greater than or equal to the image size itself.')
end

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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Correct Illumination Each Divide module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Correct Illumination Each Divide module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
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
%%% Calculates a coarse estimate of the background illumination by
%%% determining the minimum of each block in the image.  If the minimum is
%%% zero, it is recorded as .001 to prevent divide by zero errors later.
MiniIlluminationImage = blkproc(OrigImage,[BlockSize BlockSize],'max([min(x(:)); .001])');
%%% The coarse estimate is then expanded in size so that it is the same
%%% size as the original image. Bilinear interpolation is used to ensure the
%%% values do not dip below zero.
IlluminationImage1 = imresize(MiniIlluminationImage, size(OrigImage), 'bilinear');
%%% The following is used to fit a low-dimensional polynomial to the mean image.
%%% The result, IlluminationImage, is an image of the smooth illumination function.
[x,y] = meshgrid(1:size(IlluminationImage1,2), 1:size(IlluminationImage1,1));
x2 = x.*x;
y2 = y.*y;
xy = x.*y;
o = ones(size(IlluminationImage1));
Ind = find(IlluminationImage1 > 0);
Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(IlluminationImage1(Ind));
IlluminationImage2 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(IlluminationImage1));
% figure, imagesc(IlluminationImage), colormap(gray), title('Calculated illumination correction image')

%%% The final IlluminationImage is produced by dividing each
%%% pixel of the illumination image by a scalar: the minimum
%%% pixel value anywhere in the illumination image. (If the
%%% minimum value is zero, .00000001 is substituted instead.)
%%% This rescales the IlluminationImage from 1 to some number.
%%% This ensures that the final, corrected image will be in a
%%% reasonable range, from zero to 1.
IlluminationImage = IlluminationImage2 ./ min(min(IlluminationImage2));
%%% The original image is corrected based on the IlluminationImage,
%%% by dividing each pixel by the value in the IlluminationImage.
CorrectedImage = OrigImage ./ IlluminationImage;

figure, imagesc(OrigImage), title('OrigImage')
figure, imagesc(IlluminationImage), title('IlluminationImage')
figure, imagesc(IlluminationImage2), title('IlluminationImage2')

% %%% Checking to see whether the rescaling makes sense:
MAX(1) = max(max(OrigImage));
MIN(1) = min(min(OrigImage));
MAX(2) = max(max(IlluminationImage));
MIN(2) = min(min(IlluminationImage));
MAX(3) = max(max(CorrectedImage))
MIN(3) = min(min(CorrectedImage))

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
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the corrected
    %%%  image.
    subplot(2,2,2); imagesc(CorrectedImage); title('Illumination Corrected Image');
    %%% A subplot of the figure window is set to display the illumination
    %%% function image.
    subplot(2,2,3); imagesc(IlluminationImage); title('Illumination Function');
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
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Correct Illumination Divide Each Min module: 
%%%%% .
%%%%%   This module corrects for uneven illumination of each image, based on
%%%%% information contained only within that image.  It is preferable to
%%%%% use a correct illumination module that corrects for illumination
%%%%% based on all images acquired at the same time.
%%%%%    First, the minimum pixel value is determined within each
%%%%% "block" of the image.  The block dimensions are entered by the user,
%%%%% and should be large enough that every block is likely to contain some
%%%%% "background" pixels, where no cells are located.  Theoretically, the
%%%%% intensity values of these background pixels should always be the same
%%%%% number.  With uneven illumination, the background pixels will vary
%%%%% across the image, and this yields a function that presumably affects
%%%%% the intensity of the "real" pixels, those that comprise cells.
%%%%% Therefore, once the minimums are determined across the image, the
%%%%% minimums are smoothed out. This produces an image
%%%%% that represents the variation in illumination across the field of
%%%%% view.  The original image is *divided* by this image to produce the
%%%%% corrected image.
%%%%% . 
%%%%% This module is loosely based on the Matlab demo "Correction of non-uniform
%%%%% illumination" in the Image Processing Toolbox demos "Enhancement"
%%%%% category.
%%%%% MATLAB6p5/toolbox/images/imdemos/examples/enhance/ipss003.html
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