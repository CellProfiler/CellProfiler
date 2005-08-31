function handles = RGBSplit(handles)

% Help for the RGB Split module:
% Category: Image Processing
%
% Takes an RGB image and splits into three separate grayscale images.
%
% SAVING IMAGES: The three grayscale images produced by this module
% can be easily saved using the Save Images module, using the names
% you assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the section marked SAVE DATA TO HANDLES STRUCTURE) and then use
% the Save Images module.
%
% See also RGBTOGRAY, RGBMERGE.

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




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the color image to be split into grayscale images?
%infotypeVAR01 = imagegroup
RGBImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the image that was red? Type N to ignore red.
%defaultVAR02 = OrigRed
%infotypeVAR02 = imagegroup indep
RedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the image that was green? Type N to ignore green.
%defaultVAR03 = OrigGreen
%infotypeVAR03 = imagegroup indep
GreenImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the image that was blue? Type N to ignore blue.
%defaultVAR04 = OrigBlue
%infotypeVAR04 = imagegroup indep
BlueImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the RGB image from the handles structure.
fieldname = ['', RGBImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the RGB Split module could not find the input image.  It was supposed to be named ', RGBImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
RGBImage = handles.Pipeline.(fieldname);

Size = size(RGBImage);
if length(Size) ~= 3
    error('Image processing was canceled because the RGB image you specified in the RGB Split module could not be separated into three layers of image data.  Is it a color image?')
end
if Size(3) ~= 3
    error('Image processing was canceled because the RGB image you specified in the RGB Split module could not be separated into three layers of image data.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Determines whether the user has specified an image to be loaded in
%%% blue.
if ~strcmp(upper(RedImageName), 'N')
    RedImage = RGBImage(:,:,1);
else
    RedImage = zeros(size(RGBImage(:,:,1)));
end
if ~strcmp(upper(GreenImageName), 'N')
    GreenImage = RGBImage(:,:,2);
else
    GreenImage = zeros(size(RGBImage(:,:,1)));
end
if ~strcmp(upper(BlueImageName), 'N')
    BlueImage = RGBImage(:,:,3);
else
    BlueImage = zeros(size(RGBImage(:,:,1)));
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);

    %%% A subplot of the figure window is set to display the Splitd RGB
    %%% image.  Using imagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1); imagesc(RGBImage);
    title(['Input RGB Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the blue image.
    subplot(2,2,2); imagesc(BlueImage); CPcolormap(handles), title('Blue Image');
    %%% A subplot of the figure window is set to display the green image.
    subplot(2,2,3); imagesc(GreenImage); CPcolormap(handles), title('Green Image');
    %%% A subplot of the figure window is set to display the red image.
    subplot(2,2,4); imagesc(RedImage); CPcolormap(handles), title('Red Image');
    CPFixAspectRatio(RGBImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(RedImageName) = RedImage;
handles.Pipeline.(GreenImageName) = GreenImage;
handles.Pipeline.(BlueImageName) = BlueImage;
