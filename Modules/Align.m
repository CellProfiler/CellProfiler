function handles = AlgAlignImages(handles)

% Help for the Align module:
% Category: Pre-processing
% 
% For two or three input images, this module determines the optimal
% alignment among them.  This works whether the images are correlated
% or anti-correlated (bright in one = bright in the other, or bright
% in one = dim in the other).  This is useful when the microscope is
% not perfectly calibrated, because, for example, proper alignment is
% necessary for primary objects to be helpful to identify secondary
% objects. The images are cropped appropriately according to this
% alignment, so the final images will be smaller than the originals by
% a few pixels if alignment is necessary.
% 
% Which image is displayed as which color can be changed by going into
% the module's '.m' file and changing the lines after 'FOR DISPLAY
% PURPOSES ONLY'.  The first line in each set is red, then green, then
% blue.
%
% SAVING IMAGES: The three aligned images produced by this module can
% be easily saved using the Save Images module, using the names you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.
%
% See also ALGALIGNANDCROP.

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
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.

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

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the algorithm.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this algorithm with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown. 
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the first image to be aligned? (will be displayed as blue)
%defaultVAR01 = OrigBlue
Image1Name = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
AlignedImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What did you call the second image to be aligned? (will be displayed as green)
%defaultVAR03 = OrigGreen
Image2Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the aligned second image?
%defaultVAR04 = AlignedGreen
AlignedImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What did you call the third image to be aligned? (will be displayed as red)
%defaultVAR05 = /
Image3Name = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call the aligned third image?
%defaultVAR06 = /
AlignedImage3Name = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = This module calculates the alignment shift. Do you want to actually adjust the images?
%defaultVAR07 = N
AdjustImage = upper(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%%%VariableRevisionNumber = 02
% The variables have changed for this module.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(Image1Name,'/') == 1
    error('Image processing was canceled because no image was loaded in the Align module''s first image slot')
end
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, Image1Name) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image1Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
Image1 = handles.Pipeline.(Image1Name);

%%% Same for Image 2.
if strcmp(Image2Name,'/') == 1
    error('Image processing was canceled because no image was loaded in the Align module''s second image slot')
end
if isfield(handles.Pipeline, Image2Name) == 0
    error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image2Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
Image2 = handles.Pipeline.(Image2Name);

%%% Same for Image 3.
if strcmp(Image3Name,'/') ~= 1
    if isfield(handles.Pipeline, Image3Name) == 0
        error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image3Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    Image3 = handles.Pipeline.(Image3Name);
end

%%% Determine the filenames of the images to be analyzed.
fieldname = ['Filename', Image1Name];
FileName1 = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
fieldname = ['Filename', Image2Name];
FileName2 = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
if strcmp(upper(Image3Name),'/') ~= 1
    fieldname = ['Filename', Image3Name];
    FileName3 = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
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

%%% Aligns three input images.
if strcmp(Image3Name,'/') ~= 1
    %%% Aligns 1 and 2 (see subfunctions at the end of the module).
    [sx, sy] = autoalign(Image1, Image2);
    Temp1 = subim(Image1, sx, sy);
    Temp2 = subim(Image2, -sx, -sy);
    %%% Assumes 3 is stuck to 2.
    Temp3 = subim(Image3, -sx, -sy);
    %%% Aligns 2 and 3.
    [sx2, sy2] = autoalign(Temp2, Temp3);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy), ...
        ') (2 vs 3: X ', num2str(sx2), ', Y ', num2str(sy2),')'];
    if strcmp(AdjustImage,'Y') == 1
        AlignedImage2 = subim(Temp2, sx2, sy2);
        AlignedImage3 = subim(Temp3, -sx2, -sy2);
        %%% 1 was already aligned with 2.
        AlignedImage1 = subim(Temp1, sx2, sy2);
    end
else %%% Aligns two input images.
    [sx, sy] = autoalign(Image1, Image2);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy),')'];
    if strcmp(AdjustImage,'Y') == 1
        AlignedImage1 = subim(Image1, sx, sy);
        AlignedImage2 = subim(Image2, -sx, -sy);
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this module. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

%%% Determines the figure number to display in.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisAlgFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisAlgFigureNumber) == 1;
    if strcmp(AdjustImage,'Y') == 1
        %%% For three input images.
        if strcmp(Image3Name,'/') ~= 1
            OriginalRGB(:,:,1) = Image3;
            OriginalRGB(:,:,2) = Image2;
            OriginalRGB(:,:,3) = Image1;
            AlignedRGB(:,:,1) = AlignedImage3;
            AlignedRGB(:,:,2) = AlignedImage2;
            AlignedRGB(:,:,3) = AlignedImage1;
        else %%% For two input images.
            OriginalRGB(:,:,1) = zeros(size(Image1));
            OriginalRGB(:,:,2) = Image2;
            OriginalRGB(:,:,3) = Image1;
            AlignedRGB(:,:,1) = zeros(size(AlignedImage1));
            AlignedRGB(:,:,2) = AlignedImage2;
            AlignedRGB(:,:,3) = AlignedImage1;
        end
    end
    if handles.Current.SetBeingAnalyzed == 1
        %%% Sets the window to be only 250 pixels wide.
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisAlgFigureNumber, 'position', newsize);
    end
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
    if strcmp(AdjustImage,'Y') == 1
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,1,1); imagesc(OriginalRGB);
        title(['Input Images, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the adjusted
        %%%  image.
        subplot(2,1,2); imagesc(AlignedRGB); title('Aligned Images');
    end
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    set(displaytexthandle,'string',Results)
    set(ThisAlgFigureNumber,'toolbar','figure')
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
% 'structure', and whose name is handles. The contents of the handles
% structure are printed out at the command line of Matlab using the
% Tech Diagnosis button. The only variables present in the main
% handles structure are handles to figures and gui elements.
% Everything else should be saved in one of the following
% substructures:
%
% handles.Settings:
%       Everything in handles.Settings is stored when the user uses
% the Save pipeline button, and these data are loaded into
% CellProfiler when the user uses the Load pipeline button. This
% substructure contains all necessary information to re-create a
% pipeline, including which modules were used (including variable
% revision numbers), their setting (variables), and the pixel size.
%   Fields currently in handles.Settings: PixelSize, ModuleNames,
% VariableValues, NumbersOfVariables, VariableRevisionNumbers.
%
% handles.Pipeline:
%       This substructure is deleted at the beginning of the
% analysis run (see 'Which substructures are deleted prior to an
% analysis run?' below). handles.Pipeline is for storing data which
% must be retrieved by other modules. This data can be overwritten as
% each image set is processed, or it can be generated once and then
% retrieved during every subsequent image set's processing, or it can
% be saved for each image set by saving it according to which image
% set is being analyzed, depending on how it will be used by other
% modules. Any module which produces or passes on an image needs to
% also pass along the original filename of the image, named after the
% new image name, so that if the SaveImages module attempts to save
% the resulting image, it can be named by appending text to the
% original file name.
%   Example fields in handles.Pipeline: FileListOrigBlue,
% PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which contains the actual image).
%
% handles.Current:
%       This substructure contains information needed for the main
% CellProfiler window display and for the various modules to
% function. It does not contain any module-specific data (which is in
% handles.Pipeline).
%   Example fields in handles.Current: NumberOfModules,
% StartupDirectory, DefaultOutputDirectory, DefaultImageDirectory,
% FilenamesInImageDir, CellProfilerPathname, ImageToolHelp,
% DataToolHelp, FigureNumberForModule01, NumberOfImageSets,
% SetBeingAnalyzed, TimeStarted, CurrentModuleNumber.
%
% handles.Preferences: 
%       Everything in handles.Preferences is stored in the file
% CellProfilerPreferences.mat when the user uses the Set Preferences
% button. These preferences are loaded upon launching CellProfiler.
% The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
% fields can be changed for the current session by the user using edit
% boxes in the main CellProfiler window, which changes their values in
% handles.Current. Therefore, handles.Current is most likely where you
% should retrieve this information if needed within a module.
%   Fields currently in handles.Preferences: PixelSize, FontSize,
% DefaultModuleDirectory, DefaultOutputDirectory,
% DefaultImageDirectory.
%
% handles.Measurements:
%       Everything in handles.Measurements contains data specific to each
% image set analyzed for exporting. It is used by the ExportMeanImage
% and ExportCellByCell data tools. This substructure is deleted at the
% beginning of the analysis run (see 'Which substructures are deleted
% prior to an analysis run?' below).
%    Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly. It is likely that
% Subobject will become a new prefix, when measurements will be
% collected for objects contained within other objects. 
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a
% field of X locations and a field of Y locations. It is wise to
% include the user's input for 'ObjectName' or 'ImageName' as part of
% the fieldname in the handles structure so that multiple modules can
% be run and their data will not overwrite each other.
%   Example fields in handles.Measurements: ImageCountNuclei,
% ObjectAreaCytoplasm, FilenameOrigBlue, PathnameOrigBlue,
% TimeElapsed.
%
% Which substructures are deleted prior to an analysis run?
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the beginning of the analysis run, whereas
% anything stored in handles.Settings, handles.Preferences, and
% handles.Current will be retained from one analysis to the next. It
% is important to think about which of these data should be deleted at
% the end of an analysis run because of the way Matlab saves
% variables: For example, a user might process 12 image sets of nuclei
% which results in a set of 12 measurements ("ImageTotalNucArea")
% stored in handles.Measurements. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "ImageTotalNucArea"
% to analyze 4 image sets, the 4 measurements will overwrite the first
% 4 measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only a module which
% depends on the output "SegmNucImg" but does not run a module that
% produces an image by that name, the module will run just fine: it
% will just repeatedly use the processed image of nuclei leftover from
% the last image set, which was left in handles.Pipeline.

if strcmp(AdjustImage,'Y') == 1
    %%% Saves the adjusted image to the
    %%% handles structure so it can be used by subsequent modules.
    handles.Pipeline.(AlignedImage1Name) = AlignedImage1;
    handles.Pipeline.(AlignedImage2Name) = AlignedImage2;
    if strcmp(Image3Name,'/') ~= 1
        handles.Pipeline.(AlignedImage3Name) = AlignedImage3;
    end
end
%%% Saves the original file name ito the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['Filename', AlignedImage1Name];
handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName1;
fieldname = ['Filename', AlignedImage2Name];
handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName2;
if strcmp(Image3Name,'/') ~= 1
fieldname = ['Filename', AlignedImage3Name];
handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName3;
end

%%% Stores the shift in alignment as a measurement for quality control
%%% purposes.
fieldname = ['ImageXAlign', AlignedImage1Name,AlignedImage2Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sx};
fieldname = ['ImageYAlign', AlignedImage1Name,AlignedImage2Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sy};

%%% If three images were aligned:
if strcmp(Image3Name,'/') ~= 1
fieldname = ['ImageXAlignFirstTwoImages',AlignedImage3Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sx2};
fieldname = ['ImageYAlignFirstTwoImages',AlignedImage3Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sy2};
end

%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%

function [shiftx, shifty] = autoalign(in1, in2)
%%% Aligns two images using mutual-information and hill-climbing.
best = mutualinf(in1, in2);
bestx = 0;
besty = 0;
%%% Checks which one-pixel move is best.
for dx=-1:1,
  for dy=-1:1,
    cur = mutualinf(subim(in1, dx, dy), subim(in2, -dx, -dy));
    if (cur > best),
      best = cur;
      bestx = dx;
      besty = dy;
    end
  end
end
if (bestx == 0) && (besty == 0),
  shiftx = 0;
  shifty = 0;
  return;
end
%%% Remembers the lastd direction we moved.
lastdx = bestx;
lastdy = besty;
%%% Loops until things stop improving.
while true,
  [nextx, nexty, newbest] = one_step(in1, in2, bestx, besty, lastdx, lastdy, best);
  if (nextx == 0) && (nexty == 0),
    shiftx = bestx;
    shifty = besty;
    return;
  else
    bestx = bestx + nextx;
    besty = besty + nexty;
    best = newbest;
  end
end

function [nx, ny, nb] = one_step(in1, in2, bx, by, ldx, ldy, best)
%%% Finds the best one pixel move, but only in the same direction(s) we
%%% moved last time (no sense repeating evaluations)
nb = best;
for dx=-1:1,
  for dy=-1:1,
    if (dx == ldx) || (dy == ldy),
      cur = mutualinf(subim(in1, bx+dx, by+dy), subim(in2, -(bx+dx), -(by+dy)));
      if (cur > nb),
        nb = cur;
        nx = dx;
        ny = dy;
      end
    end
  end
end
if (best == nb),
  %%% no change, so quit searching
  nx = 0;
  ny = 0;
end

function sub = subim(im, dx, dy)
%%% Subimage with positive or negative offsets
if (dx > 0),
  sub = im(:,dx+1:end);
else
  sub = im(:,1:end+dx);
end
if (dy > 0),
  sub = sub(dy+1:end,:);
else
  sub = sub(1:end+dy,:);
end

function H = entropy(X)
%%% Entropy of samples X
S = imhist(X,256);
%%% if S is probability distribution function N is 1
N=sum(sum(S));
if ((N>0) && (min(S(:))>=0))
   Snz=nonzeros(S);
   H=log2(N)-sum(Snz.*log2(Snz))/N;
else
   H=0;
end

function H = entropy2(X,Y)
%%% joint entropy of paired samples X and Y
%%% Makes sure images are binned to 256 graylevels
X = double(im2uint8(X));
Y = double(im2uint8(Y));
%%% Creates a combination image of X and Y
XY = 256*X + Y;
S = histc(XY(:),0:(256*256-1));
%%% If S is probability distribution function N is 1
N=sum(sum(S));          
if ((N>0) && (min(S(:))>=0))
   Snz=nonzeros(S);
   H=log2(N)-sum(Snz.*log2(Snz))/N;
else
   H=0;
end

function I = mutualinf(X, Y)
%%% Mutual information of images X and Y
I = entropy(X) + entropy(Y) - entropy2(X,Y);