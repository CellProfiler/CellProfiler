function handles = AlgCorrectIllumSubtractAllMin(handles)

% Help for the Correct Illumination Subtract All Min module: 
% Category: Pre-processing
% 
% This module corrects for uneven illumination of each image, based on
% information from a set of images collected at the same time. 
%
% How it works:
% First, the minimum pixel value is determined within each "block" of
% each image, and the values are averaged together for all images.
% The block dimensions are entered by the user, and should be large
% enough that every block is likely to contain some "background"
% pixels, where no cells are located. Theoretically, the intensity
% values of these background pixels should always be the same number.
% With uneven illumination, the background pixels will vary across the
% image, and this yields a function that presumably affects the
% intensity of the "real" pixels, those that comprise cells.
% Therefore, once the average minimums are determined across the
% images, the minimums are smoothed out. This produces an image that
% represents the variation in illumination across the field of view.
% This process is carried out before the first image set is processed.
% This image is then subtracted from each original image to produce
% the corrected image.
% 
% If you want to run this module only to calculate the mean and
% illumination images and not to correct every image in the directory,
% simply run the module as usual and use the button on the Timer to
% stop processing after the first image set.
%
% SAVING IMAGES: The illumination corrected images produced by this
% module can be easily saved using the Save Images module, using the
% name you assign. The mean image can be saved using the name
% MeanIlluminationImageAS plus whatever you called the corrected image
% (e.g. MeanIlluminationImageASCorrBlue). The Illumination correction
% image can be saved using the name IllumImageAS plus whatever you
% called the corrected image (e.g. IllumImageASCorrBlue).  Note that
% using the Save Images module saves a copy of the image in an image
% file format, which has lost some of the detail that a matlab file
% format would contain.  In other words, if you want to save the
% illumination image to use it in a later analysis, you should use the
% settings boxes within this module to save the illumination image in
% '.mat' format. If you want to save other intermediate images, alter
% the code for this module to save those images to the handles
% structure (see the SaveImages module help) and then use the Save
% Images module.
%
% See also ALGCORRECTILLUMDIVIDEALLMEANRETRIEVEIMG,
% ALGCORRECTILLUMDIVIDEALLMEAN,
% ALGCORRECTILLUMDIVIDEEACHMIN_9, ALGCORRECTILLUMDIVIDEEACHMIN_10,
% ALGCORRECTILLUMSUBTRACTEACHMIN.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
% 
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
% 
% Authors:
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

%textVAR03 = Block size. This should be set large enough that every square block 
%textVAR04 = of pixels is likely to contain some background.
%defaultVAR04 = 60
BlockSize = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR08 = To save the illum. corr. image to use later, type a file name + .mat. Else, 'N'
%defaultVAR08 = N
IllumCorrectFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = If you have already created an illumination corrrection image to be used, enter the 
%textVAR10 = path & file name of the image below. To calculate the illumination correction image 
%textVAR11 = from all the images of this color that will be processed, leave a slash in the box below.#LongBox#
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


%%% Checks whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
MinLengthWidth = min(m,n);
if BlockSize >= MinLengthWidth
        error('Image processing was canceled because in the Correct Illumination module the selected block size is greater than or equal to the image size itself.')
end

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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
            set(h, 'Position', PositionMsgBox)
            drawnow
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
            [dum,ndx] = min(ceil(m./MM).*MM-m); %#ok We want to ignore MLint error checking for this line.
            BestBlockSize(1) = MM(ndx);
            [dum,ndx] = min(ceil(n./NN).*NN-n); %#ok We want to ignore MLint error checking for this line.
            BestBlockSize(2) = NN(ndx);
            BestRows = BestBlockSize(1)*ceil(m/BestBlockSize(1));
            BestColumns = BestBlockSize(2)*ceil(n/BestBlockSize(2));
            RowsToAdd = BestRows - m;
            ColumnsToAdd = BestColumns - n;
            
            %%% Calculates a coarse estimate of the background illumination by
            %%% determining the minimum of each block in the image.
            MiniIlluminationImage = blkproc(padarray(im2double(imread(char(FileList(1)))),[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(:))');
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
    if exist('MeanIlluminationImage','var') == 1
        fieldname = ['MeanIlluminationImageAS', CorrectedImageName];
        handles.Pipeline.(fieldname) = MeanIlluminationImage;        
    end
    fieldname = ['IllumImageAS', CorrectedImageName];
    handles.Pipeline.(fieldname) = IlluminationImage;
end

%%% The following is run for every image set. Retrieves the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['MeanIlluminationImageAS', CorrectedImageName];
if isfield(handles.Pipeline, fieldname) == 1
    MeanIlluminationImage = handles.Pipeline.(fieldname);
end
fieldname = ['IllumImageAS', CorrectedImageName];
IlluminationImage = handles.Pipeline.(fieldname);
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
    if exist('MeanIlluminationImage','var') == 1
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
% handles.Pipeline is for storing data which must be retrieved by other modules.
% This data can be overwritten as each image set is processed, or it
% can be generated once and then retrieved during every subsequent image
% set's processing, or it can be saved for each image set by
% saving it according to which image set is being analyzed.
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the end of the analysis run, whereas anything
% stored in handles.Settings will be retained from one analysis to the
% next. It is important to think about which of these data should be
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
%       Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly.
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%       Extracting measurements: handles.Measurements.CenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.Measurements.AreaNuclei{2}(1) gives the area of the first object in
% the second image.

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent algorithms.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a field named
%%% after the corrected image name.
fieldname = ['Filename', CorrectedImageName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;