function handles = ImageMath(handles)

% Help for the ImageMath module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Performs simple mathematical operations on image intensities.
% *************************************************************************
%
% Operation:
% The complement of a grayscale image inverts the intensities 
% (dark areas become brighter, and bright areas become darker).  
%
% Average in the ImageMath module is the numerical average of the two images loaded in
% the module.  If you would like to average many images (all of the images
% in an entire pipeline), please use the CorrectIllumination_Calculate
% module and chose the option "(For 'All' mode only) What do you want to
% call the averaged image (prior to dilation or smoothing)? 
% (This is an image produced during the calculations - it is typically not
% needed for downstream modules)"  This will be an average over all images.
% 
% Multiply factors:
% The final image may have a substantially different range of pixel
% intensities than the originals, so each image can be multiplied by a 
% factor prior to the operation. This factor can be any real number.
%
% Do you want values in the image to be set to zero/one?:
% Values outside the range of 0 to 1 might not be handled well by other
% modules. Here, you have the option of setting negative values to 0.
% For other options (e.g. setting values over 1 to equal 1), see the
% Rescale Intensity module.
%
% See also SubtractBackground, RescaleIntensity.

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

%textVAR01 = What do you want to call the resulting image?
%defaultVAR01 = ImageAfterMath
%infotypeVAR01 = imagegroup indep
ImageAfterMathName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Choose first image:
%infotypeVAR02 = imagegroup
FirstImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Choose second image, or "Other..." and enter a constant. Note: if the operation chosen below is 'Complement' or 'Log transform', this second image will not be used
%infotypeVAR03 = imagegroup
SecondImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = What operation would you like performed?
%choiceVAR04 = Add
%choiceVAR04 = Subtract
%choiceVAR04 = Multiply
%choiceVAR04 = Divide
%choiceVAR04 = Complement
%choiceVAR04 = Log transform (base 2)
%choiceVAR04 = Average
Operation = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Enter a factor to multiply the first image by (before other operations):
%defaultVAR05 = 1
MultiplyFactor1 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter a factor to multiply the second image by (before other operations):
%defaultVAR06 = 1
MultiplyFactor2 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Enter an exponent to raise the result to *after* chosen operation:
%defaultVAR07 = 1
Power = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Enter a factor to multipy the result by *after* chosen operation:
%defaultVAR08 = 1
MultiplyFactor3 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = Do you want negative values in the image to be set to zero?
%choiceVAR09 = Yes
%choiceVAR09 = No
FloorZero = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR09 = Do you want values greater than one in the image to be set to one?
%choiceVAR09 = Yes
%choiceVAR09 = No
CeilingOne = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% Check for user-set constants
if ~isempty(str2num(FirstImageName))
    error(['Choosing a number for the first image name within ' ModuleName ...
        ' is not allowed.  If you need to access a user-defined image name,'...
        'please enter an image previously defined.'])
end
SecondImageConstant = str2num(SecondImageName);    %% This will be empty unless a constant was input

%%% Reads (opens) the images you want to analyze and assigns them to
%%% variables.
FirstImage = CPretrieveimage(handles,FirstImageName,ModuleName,'MustBeGray','CheckScale');
if isempty(SecondImageConstant)
    SecondImage = CPretrieveimage(handles,SecondImageName,ModuleName,'MustBeGray','CheckScale');
else 
    SecondImage = SecondImageConstant;
    clear SecondImageConstant
end

%%% Check to make sure multiply factors are valid entries. If not change to
%%% default and warn user.
if isnan(MultiplyFactor1)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': First multiply factor invalid']))
        CPwarndlg(['The first image multiply factor you have entered in the ', ModuleName, ' module is invalid, it is being reset to 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': First multiply factor invalid'],'replace');
    end
    MultiplyFactor1 = 1;
end
if isnan(MultiplyFactor2)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Second multiply factor invalid']))
        CPwarndlg(['The second image multiply factor you have entered in the ', ModuleName, ' module is invalid, it is being reset to 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Second multiply factor invalid'],'replace');
    end
    MultiplyFactor2 = 1;
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

switch Operation
    case 'Add'
        ImageAfterMath = imadd(MultiplyFactor1*FirstImage,MultiplyFactor2*SecondImage);
    case 'Subtract'
        ImageAfterMath = imsubtract(MultiplyFactor1*FirstImage,MultiplyFactor2*SecondImage);
    case 'Multiply'
        ImageAfterMath = immultiply(MultiplyFactor1*FirstImage,MultiplyFactor2*SecondImage);
    case 'Divide'
        ImageAfterMath = imdivide(MultiplyFactor1*FirstImage,MultiplyFactor2*SecondImage);
    case 'Complement'
        ImageAfterMath = imcomplement(MultiplyFactor1*FirstImage);
    case 'Log transform (base 2)'
        if max(FirstImage(:)) > 0,
            ImageNoZeros = max(FirstImage, min(FirstImage(FirstImage > 0)));
            ImageAfterMath = log2(ImageNoZeros);
        else
            ImageAfterMath = zeros(size(FirstImage));
        end
    case 'Average'
       TotalImage = imadd(MultiplyFactor1*FirstImage,MultiplyFactor2*SecondImage);
       ImageAfterMath = TotalImage/2;
end

if ~isnan(Power)
    ImageAfterMath = ImageAfterMath .^ Power;
end

if ~isnan(MultiplyFactor3)
    ImageAfterMath = ImageAfterMath.*MultiplyFactor3;
end

%% Apply thresholds
if strcmpi(FloorZero,'Yes')
    ImageAfterMath(ImageAfterMath < 0) = 0;
end
if strcmpi(CeilingOne,'Yes')
    ImageAfterMath(ImageAfterMath > 1) = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    CPfigure(handles,'Image',ThisModuleFigureNumber);

    %% NumColumns is useful since 'Complement' has only one "before" image  
    if ~strcmp(Operation,'Complement')
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(FirstImage,'TwoByTwo',ThisModuleFigureNumber);
        end
        NumColumns = 2; 
    else
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(FirstImage,'TwoByOne',ThisModuleFigureNumber);
        end
        NumColumns = 1;
    end
    
    %% Set title text
    if MultiplyFactor1 == 1
        FirstText = (FirstImageName);
    else 
        FirstText = [FirstImageName '*' num2str(MultiplyFactor1)];
    end
    if MultiplyFactor2 == 1,
        SecondText = (SecondImageName);
    else 
        SecondText = [SecondImageName '*' num2str(MultiplyFactor2)];
    end
    
    %%% First image subplot
    hAx=subplot(2,NumColumns,1,'Parent',ThisModuleFigureNumber);
    CPimagesc(MultiplyFactor1*FirstImage,handles,hAx); 
    title(hAx,[FirstText ' image, cycle # ' num2str(handles.Current.SetBeingAnalyzed)]);

    if ~strcmp(Operation,'Complement')
        %%% Second image subplot
        hAx=subplot(2,NumColumns,3,'Parent',ThisModuleFigureNumber);
        CPimagesc(MultiplyFactor2*SecondImage,handles,hAx);
        title(hAx,[SecondText ' image']);
    end
    
    %% ImageAfterMath
    hAx=subplot(2,NumColumns,2,'Parent',ThisModuleFigureNumber);
    CPimagesc(ImageAfterMath,handles,hAx);
    if ~strcmp(Operation,'Complement')
        title(hAx,[FirstText ' ' Operation ' ' SecondText ' = ' ImageAfterMathName]);
    else
        title(hAx,[FirstText ' ' Operation ' = ' ImageAfterMathName]);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure.
handles.Pipeline.(ImageAfterMathName) = ImageAfterMath;