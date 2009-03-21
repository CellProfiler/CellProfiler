function handles = RescaleIntensity(handles)

% Help for the Rescale Intensity module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Changes intensity range of an image to desired specifications.
% *************************************************************************
%
% The intensity of the incoming images are rescaled by one of several
% methods. This is especially helpful for converting 12-bit images saved in
% 16-bit format to the correct range (see method E).
%
% Settings:
%
% Rescaling method:
% (S) Stretch the image so that the minimum is zero and the maximum is
% one.
%
% (E) Enter the minimum and maximum values of the original image and the
% desired resulting image. Pixels are scaled from their user-specified 
% original range to a new user-specified range.  If the user enters "AE" 
% (Automatic for Each), then the highest and lowest pixel values will be 
% Automatically computed for each image by taking the maximum and minimum 
% pixel values in each image.  If the user enters "AA" (Automatic for All),
% then the highest and/or lowest pixel values will be Automatically computed 
% by taking the maximum and minimum pixel values in all the images in the 
% set. 
%
% The user also has the option of selecting the values that pixels 
% outside the original min/max range are set to, by entering numbers in
% the "For pixels above/below the chosen value..." boxes. If you want
% these pixels to be set to the highest/lowest rescaled intensity values, 
% enter the same number in these boxes as was entered in the highest/lowest
% rescaled intensity boxes. However, using other values permits a simple form of
% thresholding (e.g., setting the upper bounding value to 0 can be used for
% removing bright pixels above a specified value)
%
% To convert 12-bit images saved in 16-bit format to the correct 
% range, use the settings 0, 0.0625, 0, 1, 0, 1.  The value 0.0625 is equivalent 
% to 2^12 divided by 2^16, so it will convert a 16 bit image containing 
% only 12 bits of data to the proper range.
%
% (G) Rescale the image so that all pixels are equal to or greater
% than one.
%
% (M) Match the maximum of one image to the maximum of another.
%
% (C) Convert to 8 bit: Images in CellProfiler are normally stored as
% numerical class double in the range of 0 to 1. This option converts these
% images to class uint8, meaning an 8 bit integer in the range of 0 to 255.
% This is useful to reduce the amount of memory required to store the
% image. Warning: Most CellProfiler modules require the incoming image to
% be in the standard 0 to 1 range, so this conversion may cause downstream
% modules to behave unexpectedly.
%
% (T) Text: rescale by dividing by a value loaded from a text file with LoadText.
%
% See also SubtractBackground.

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

% New Settings for PyCP
% In general, this module is very confusing with the (Method E only) and
% (Method T) only etc qualifiers.  In the PyCp version, it would be great
% if the (Method E only) variables only became visible once the user had
% selected method E.  
% While I like the that Var 3 explains to the user (briefly) what each
% method does, in most modules we leave this explanation to the help to
% avoid overly long variable settings. (Think: IdPrimAuto etc)  I think it
% is fine to do the same and chane Var 3 to: 'Please select rescaling method
% below.'  If it is possible, perhaps the user could mouse over the choice
% and see a brief explanation?
%
% Vars 4&5:  This is seems more suited to a popup, with the choices being:
% AA (or 'Calculated-Automatic' or 'Calculated-All;), AE (or maybe 'Calculated-Independent' or
% 'Calculated-Each') , and 'Other...' (custom value).  A better variable description
% might be:
% 'What is the intensity from the original image that will be set to the
% lowest value in the rescaled image?
% choice1: Calculated- All
% choice2: Calculated- Each
% choice3: Other...
% If Python allows, in this an other 'popupcustom'-type cases, I almost
% think it is most obvious to leave a blank textbox for editing, rather
% than forcing the user to click other and then enter a value (since it may
% not be inherently obvious you can enter your own value?)
%
% Vars 6-9: If it's possible, condense these variables into just two:
% Var 6: Map low intensity or negative pixels from the original image to this value: (default 0)
% Var 7: Map high intensity or >1 pixels from the original image to this value: (default 1)
% I can't think of a case where you would want these to be different
% values??  I'm not sure if this is the correct way to re-word these
% variables, but I don't think they are very clear how they are presented
% now.


drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be rescaled?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the rescaled image?
%defaultVAR02 = RescaledBlue
%infotypeVAR02 = imagegroup indep
RescaledImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Rescaling method. (S) Stretch the image (0 to 1). (E) Enter the minimum and maximum values in the boxes below. (G) rescale so all pixels are equal to or Greater than one. (M) Match the maximum of one image to the maximum of another. (C) Convert to 8 bit. (T) Divide by loaded text value.  See the help for details.
%choiceVAR03 = Stretch 0 to 1
%choiceVAR03 = Enter min/max below
%choiceVAR03 = Greater than one
%choiceVAR03 = Match Maximum
%choiceVAR03 = Convert to 8 bit
%choiceVAR03 = Text: Divide by loaded text value.
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,3});
RescaleOption = RescaleOption(1);
%inputtypeVAR03 = popupmenu

%textVAR04 = (Method E only): Enter the intensity from the original image that should be set to the lowest value in the rescaled image, or type AA to calculate the lowest intensity automatically from all of the images to be analyzed and AE to calculate the lowest intensity from each image independently.
%defaultVAR04 = AA
LowestPixelOrig = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = (Method E only): Enter the intensity from the original image that should be set to the highest value in the rescaled image, or type AA to calculate the highest intensity automatically from all of the images to be analyzed and AE to calculate the highest intensity from each image independently.
%defaultVAR05 = AA
HighestPixelOrig = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = (Method E only): What value should pixels at the low end of the original intensity range be mapped to (range [0,1])?
%defaultVAR06 = 0
LowestPixelRescale = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = (Method E only): What value should pixels at the high end of the original intensity range be mapped to (range [0,1])?
%defaultVAR07 = 1
HighestPixelRescale = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = (Method E only): What value should pixels *below* the low end of the original intensity range be mapped to (range [0,1])?
%defaultVAR08 = 0
LowestPixelOrigPinnedValue = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = (Method E only): What value should pixels *above* the high end of the original intensity range be mapped to (range [0,1])?
%defaultVAR09 = 1
HighestPixelOrigPinnedValue = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = (Method M only): What did you call the image whose maximum you want the rescaled image to match?
%infotypeVAR10 = imagegroup
OtherImageName = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = (Method T only): What did you call the loaded text in the LoadText module?
%infotypeVAR11 = datagroup
TextName = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu


%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
if strncmpi(RescaleOption,'S',1)
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'M',1)
    %%% Reads (opens) the image to be analyzed and assigns it to a variable,
    %%% "MethodSpecificArguments".
    MethodSpecificArguments = CPretrieveimage(handles,OtherImageName,ModuleName,'DontCheckColor','CheckScale');
elseif strncmpi(RescaleOption,'G',1)
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'E',1)
    MethodSpecificArguments{1} = LowestPixelOrig;
    MethodSpecificArguments{2} = HighestPixelOrig;
    MethodSpecificArguments{3} = LowestPixelRescale;
    MethodSpecificArguments{4} = LowestPixelOrigPinnedValue;
    MethodSpecificArguments{5} = HighestPixelRescale;
    MethodSpecificArguments{6} = HighestPixelOrigPinnedValue;
    MethodSpecificArguments{7} = ImageName;
elseif strncmpi(RescaleOption,'C',1)
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption, 'T', 1)
    MethodSpecificArguments = TextName;
end

%%% Uses a CellProfiler subfunction.
[handles,RescaledImage] = CPrescale(handles,OrigImage,RescaleOption,MethodSpecificArguments);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber); 
    CPimagesc(OrigImage,handles,hAx);
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Rescaled
    %%% Image.
    hAx=subplot(2,1,2,'Parent',ThisModuleFigureNumber); 
    CPimagesc(RescaledImage,handles,hAx); 
    title(hAx,'Rescaled Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The Rescaled image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(RescaledImageName) = RescaledImage;