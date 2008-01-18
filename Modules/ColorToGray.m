function handles = ColorToGray(handles)

% Help for the Color To Gray module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Converts RGB (Red, Green, Blue) color images to grayscale. All channels
% can be merged into one grayscale image (COMBINE option) or each channel 
% can be extracted into a separate grayscale image (SPLIT option).
% *************************************************************************
% Note: this module is especially helpful because all identify modules
% require grayscale images.
%
% Settings:
%
% Split:
% Takes a color image and splits the three channels (red, green, blue) into
% three separate grayscale images.
%
% Combine:
% Takes a color image and converts it to grayscale by combining the three
% channels (red, green, blue) together.
%
% Adjustment factors: Leaving the adjustment factors set to 1 will balance
% all three colors equally in the final image, which will use the same
% range of intensities as the incoming image.  To weight colors relative to
% each other, the adjustment factor can be increased (to increase the
% weighting) or decreased (to decrease the weighting).
%
% See also GrayToColor.

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

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be converted to Gray?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How do you want to convert the color image?
%choiceVAR02 = Combine
%choiceVAR02 = Split
GrayOrSplit = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = COMBINE options:

%textVAR04 = What do you want to call the resulting grayscale image?
%defaultVAR04 = OrigGray
%infotypeVAR04 = imagegroup indep
GrayscaleImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the relative contribution of the red channel
%defaultVAR05 = 1
RedIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the relative contribution of the green channel
%defaultVAR06 = 1
GreenIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Enter the relative contribution of the blue channel
%defaultVAR07 = 1
BlueIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = SPLIT options:

%textVAR09 = What do you want to call the image that was red? Type N to ignore red.
%defaultVAR09 = OrigRed
%infotypeVAR09 = imagegroup indep
RedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What do you want to call the image that was green? Type N to ignore green.
%defaultVAR10 = OrigGreen
%infotypeVAR10 = imagegroup indep
GreenImageName = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What do you want to call the image that was blue? Type N to ignore blue.
%defaultVAR11 = OrigBlue
%infotypeVAR11 = imagegroup indep
BlueImageName = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS%%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(GrayOrSplit,'Combine')
    %%% Converts Image to Gray
    GrayscaleImage = (OrigImage(:,:,1)*RedIntensity+OrigImage(:,:,2)*GreenIntensity+OrigImage(:,:,3)*BlueIntensity)/(RedIntensity+GreenIntensity+BlueIntensity);
elseif strcmp(GrayOrSplit,'Split')
    %%% Determines whether the user has specified an image to be loaded in
    %%% red.
    if ~strcmp(upper(RedImageName), 'N')
        RedImage = OrigImage(:,:,1);
    else
        RedImage = zeros(size(OrigImage(:,:,1)));
    end
    if ~strcmp(upper(GreenImageName), 'N')
        GreenImage = OrigImage(:,:,2);
    else
        GreenImage = zeros(size(OrigImage(:,:,1)));
    end
    if ~strcmp(upper(BlueImageName), 'N')
        BlueImage = OrigImage(:,:,3);
    else
        BlueImage = zeros(size(OrigImage(:,:,1)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if strcmp(GrayOrSplit,'Combine')
    if any(findobj == ThisModuleFigureNumber)
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
        end
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,1,1);
        CPimagesc(OrigImage,handles);
        title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the Grayscale
        %%% Image.
        subplot(2,1,2);
        CPimagesc(GrayscaleImage,handles);
        title('Grayscale Image');
    end
elseif strcmp(GrayOrSplit,'Split')
    if any(findobj == ThisModuleFigureNumber)
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
        end
        %%% A subplot of the figure window is set to display the Splitd RGB
        %%% image.  Using CPimagesc or image instead of imshow doesn't work when
        %%% some of the pixels are saturated.
        subplot(2,2,1);
        CPimagesc(OrigImage,handles);
        title(['Input Color Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the blue image.
        subplot(2,2,2);
        CPimagesc(BlueImage,handles);
        title('Blue Image');
        %%% A subplot of the figure window is set to display the green image.
        subplot(2,2,3);
        CPimagesc(GreenImage,handles);
        title('Green Image');
        %%% A subplot of the figure window is set to display the red image.
        subplot(2,2,4);
        CPimagesc(RedImage,handles);
        title('Red Image');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(GrayOrSplit,'Combine')
    %%% Saves the Grayscaled image to the handles structure so it can be
    %%% used by subsequent modules.
    fieldname = GrayscaleImageName;
    handles.Pipeline.(fieldname) = GrayscaleImage;
elseif strcmp(GrayOrSplit,'Split')
    %%% Saves the adjusted image to the handles structure so it can be used by
    %%% subsequent modules.
    handles.Pipeline.(RedImageName) = RedImage;
    handles.Pipeline.(GreenImageName) = GreenImage;
    handles.Pipeline.(BlueImageName) = BlueImage;
end