function handles = AlgCorrectIllumSubtractEachMin(handles)

% Help for the Correct Illumination Subtract Each Min module: 
% 
% This module corrects for uneven illumination of each image, based on
% information contained only within that image.  It is preferable to
% use a correct illumination module that corrects for illumination
% based on all images acquired at the same time.
%
% First, the minimum pixel value is determined within each
% "block" of the image.  The block dimensions are entered by the user,
% and should be large enough that every block is likely to contain some
% "background" pixels, where no cells are located.  Theoretically, the
% intensity values of these background pixels should always be the same
% number.  With uneven illumination, the background pixels will vary
% across the image, and this yields a function that presumably affects
% the intensity of the "real" pixels, those that comprise cells.
% Therefore, once the minimums are determined across the image, the
% minimums are smoothed out. This produces an image that represents the
% variation in illumination across the field of view.  This image is
% then subtracted from the original image to produce the corrected
% image.
% 
% This module does not rescale or otherwise adjust the resulting image,
% so that intensity measurements will be accurate, assuming that all
% images should have the same background levels (at spots where no
% cells are located).
% 
% This module is based on the Matlab demo "Correction of non-uniform
% illumination" in the Image Processing Toolbox demos "Enhancement"
% category.
% MATLAB6p5/toolbox/images/imdemos/examples/enhance/ipss003.html

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
% The Original Code is the Correct Illumination Subtract Each Minimum module.
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

%textVAR03 = Block size. This should be set large enough that every square block 
%textVAR04 = of pixels is likely to contain some background.
%defaultVAR04 = 60
BlockSize = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

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
%%% Calculates a coarse estimate of the background illumination by
%%% determining the minimum of each block in the image.
MiniIlluminationImage = blkproc(OrigImage,[BlockSize BlockSize],'min(x(:))');
drawnow
%%% The coarse estimate is then expanded in size so that it is the same
%%% size as the original image. Bicubic 
%%% interpolation is used to ensure that the data is smooth.
IlluminationImage1 = imresize(MiniIlluminationImage ,size(OrigImage), 'bicubic');
%%% The following is used to fit a low-dimensional polynomial to the mean image.
%%% The result, IlluminationImage, is an image of the smooth illumination function.
drawnow
[x,y] = meshgrid(1:size(IlluminationImage1,2), 1:size(IlluminationImage1,1));
x2 = x.*x;
y2 = y.*y;
xy = x.*y;
o = ones(size(IlluminationImage1));
Ind = find(IlluminationImage1 > 0);
Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(IlluminationImage1(Ind));
IlluminationImage = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(IlluminationImage1));
drawnow
%%% The background at each pixel is subtracted from the original image.
CorrectedImage = imsubtract(OrigImage, IlluminationImage);
drawnow
%%% Converts negative values to zero.  I have essentially truncated the
%%% data at zero rather than trying to rescale the data, because negative
%%% values should be fairly rare, since the minimum is used to calculate
%%% the IlluminationImage.
CorrectedImage(CorrectedImage < 0) = 0;

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
%%% Saves the original file name ito the handles structure in a
%%% field named after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;