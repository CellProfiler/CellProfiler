function handles = AlgCorrectIllumSubtractAllMin(handles)

% Help for the Correct Illumination Subtract All Min module: 
% 
% This module corrects for uneven illumination of each image, based on
% information from a set of images collected at the same time. First, the
% minimum pixel value is determined within each "block" of each image, and
% the values are averaged together for all images.  The block dimensions
% are entered by the user, and should be large enough that every block is
% likely to contain some "background" pixels, where no cells are located.
% Theoretically, the intensity values of these background pixels should
% always be the same number.  With uneven illumination, the background
% pixels will vary across the image, and this yields a function that
% presumably affects the intensity of the "real" pixels, those that
% comprise cells. Therefore, once the average minimums are determined
% across the images, the minimums are smoothed out. This produces an image
% that represents the variation in illumination across the field of view.
% This process is carried out before the first image set is processed. This
% image is then subtracted from each original image to produce the
% corrected image.
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
% The Original Code is the Correct Illumination Subtract All Minimum module.
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
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
CorrectedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Block size. This should be set large enough that every square block 
%textVAR04 = of pixels is likely to contain some background.
%defaultVAR04 = 60
BlockSize = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

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

%%% Checks whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
MinLengthWidth = min(m,n);
if BlockSize >= MinLengthWidth
        error('Image processing was canceled because in the Correct Illumination module the selected block size is greater than or equal to the image size itself.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
%%% Makes note of the current directory so the module can return to it
%%% at the end of this module.
CurrentDirectory = cd;

%%% The first time the module is run, calculates or retrieves the image to
%%% be used for correction.
if handles.setbeinganalyzed == 1
    %%% If the user has specified a path and file name of an illumination
    %%% correction image that has already been created, the image is
    %%% loaded.
    if strcmp(IllumCorrectPathAndFileName, '/') ~= 1
        try StructureIlluminationImage = load(IllumCorrectPathAndFileName);
            IlluminationImage = StructureIlluminationImage.IlluminationImage;
        catch error(['Image processing was canceled because there was a problem loading the image ', IllumCorrectPathAndFileName, '. Check that the full path and file name has been typed correctly.'])
        end
        %%% Otherwise, calculates the illumination correction image is based on all
        %%% the images of this type that will be processed.
    else 
        try
            %%% Notifies the user that the first image set will take much longer than
            %%% subsequent sets. 
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            h = msgbox('Preliminary calculations are under way for the Correct Illumination All Subtract module.  Subsequent image sets will be processed more quickly than the first image set.');
            set(h, 'Position', [PositionMsgBox])
            drawnow
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
            
            %%% Calculates the best block size that minimizes padding with
            %%% zeros, so that the illumination function will not have dim
            %%% artifacts at the right and bottom edges. (Based on Matlab's
            %%% bestblk function, but changing the minimum of the range
            %%% searched to be 75% of the suggested block size rather than
            %%% 50%.   
            %%% Defines acceptable block sizes.  m and n were
            %%% calculated above as the size of the original image.
            MM = floor(BlockSize):-1:floor(min(ceil(m/10),ceil(BlockSize*3/4)));
            NN = floor(BlockSize):-1:floor(min(ceil(n/10),ceil(BlockSize*3/4)));
            %%% Chooses the acceptable block that has the minimum padding.
            [dum,ndx] = min(ceil(m./MM).*MM-m); 
            BestBlockSize(1) = MM(ndx);
            [dum,ndx] = min(ceil(n./NN).*NN-n); 
            BestBlockSize(2) = NN(ndx);
            BestRows = BestBlockSize(1)*ceil(m/BestBlockSize(1));
            BestColumns = BestBlockSize(2)*ceil(n/BestBlockSize(2));
            RowsToAdd = BestRows - m;
            ColumnsToAdd = BestColumns - n;
            
            %%% Calculates a coarse estimate of the background illumination by
            %%% determining the minimum of each block in the image.
            MiniIlluminationImage = blkproc(padarray(im2double(imread(char(FileList(1)))),[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(:))');
            % figure, imshow(MiniIlluminationImage), title('first image')
            for i=2:length(FileList)
                MiniIlluminationImage = MiniIlluminationImage + blkproc(padarray(im2double(imread(char(FileList(i)))),[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(:))');
            end
            MeanMiniIlluminationImage = MiniIlluminationImage / length(FileList);
            %%% Expands the coarse estimate in size so that it is the same
            %%% size as the original image. Bicubic 
            %%% interpolation is used to ensure that the data is smooth.
            MeanIlluminationImage = imresize(MeanMiniIlluminationImage, size(im2double(imread(char(FileList(1))))), 'bicubic');
            %%% Fits a low-dimensional polynomial to the mean image.
            %%% The result, IlluminationImage, is an image of the smooth illumination function.
            [x,y] = meshgrid(1:size(MeanIlluminationImage,2), 1:size(MeanIlluminationImage,1));
            x2 = x.*x;
            y2 = y.*y;
            xy = x.*y;
            o = ones(size(MeanIlluminationImage));
            Ind = find(MeanIlluminationImage > 0);
            Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(MeanIlluminationImage(Ind));
            IlluminationImage = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(MeanIlluminationImage));
            %%% Note: the following "imwrite" saves the illumination
            %%% correction image in TIF format, but the image is compressed
            %%% so it is not as smooth as the image that is saved using the
            %%% "save" function below, which is stored in matlab ".mat"
            %%% format.
            % imwrite(IlluminationImage, 'IlluminationImage.tif', 'tif')
            
            %%% Saves the illumination correction image to the hard
            %%% drive if requested.
            if strcmp(IllumCorrectFileName, 'N') == 0
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
    if exist('MeanIlluminationImage') == 1
        fieldname = ['dOTMeanIlluminationImageAS', ImageName];
        handles.(fieldname) = MeanIlluminationImage;        
    end
    fieldname = ['dOTIllumImageAS', ImageName];
    handles.(fieldname) = IlluminationImage;
end

%%% The following is run for every image set. Retrieves the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['dOTMeanIlluminationImageAS', ImageName];
if isfield(handles, fieldname) == 1
    MeanIlluminationImage = handles.(fieldname);
end
fieldname = ['dOTIllumImageAS', ImageName];
IlluminationImage = handles.(fieldname);
%%% Corrects the original image based on the IlluminationImage,
%%% by subtracting each pixel by the value in the IlluminationImage.
CorrectedImage = imsubtract(OrigImage, IlluminationImage);

%%% Converts negative values to zero.  I have essentially truncated the
%%% data at zero rather than trying to rescale the data, because negative
%%% values should be fairly rare (and minor), since the minimum is used to
%%% calculate the IlluminationImage.
CorrectedImage(CorrectedImage < 0) = 0;

%%% Returns to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
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
    if exist('MeanIlluminationImage') == 1
        subplot(2,2,3); imagesc(MeanIlluminationImage); 
        title(['Mean Illumination in all ', ImageName, ' images']);
    end
    subplot(2,2,4); imagesc(IlluminationImage); 
    title('Illumination Function'); colormap(gray)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent algorithms.
fieldname = ['dOT', CorrectedImageName];
handles.(fieldname) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a field named
%%% after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;