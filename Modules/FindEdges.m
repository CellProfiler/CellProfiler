function handles = FindEdges(handles)

% Help for the Flip module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Flips an image from top to bottom, left to right, or both.
% *************************************************************************

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image you want to find the edges of?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call image?
%defaultVAR02 = EdgedImaged
%infotypeVAR02 = imagegroup indep
OutputName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What method would you like to use?
%choiceVAR03 = Roberts
%choiceVAR03 = Sobel
%choiceVAR03 = Prewitt
%choiceVAR03 = Log
%choiceVAR03 = Canny
%inputtypeVAR03 = popupmenu
Method = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want the threshold to be?  Put in a '/' to have one picked automatically. 
%defaultVAR04 = /
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = For the Roberts method, do you want edge thinning?
%choiceVAR05 = Thinning
%choiceVAR05 = No Thinning
%inputtypeVAR05 = popupmenu
Thinning = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For the Sobel and Prewitt methods, which direction?
%choiceVAR06 = Both
%choiceVAR06 = Horizontal
%choiceVAR06 = Vertical
%inputtypeVAR06 = popupmenu
Direction = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = For the log and Canny methods, what is the value of sigma? Use '/' for defaults.
%defaultVAR07 = /
Sigma = str2double(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For the Canny method, what is the low threshold? Use '/' for defaults.
%defaultVAR08 = /
CannyLowThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));


%textVAR09 = What is the Threshold Correction Factor? The Threshold can be automatically found, then multiplied by this factor.
%defaultVAR09 = 1
ThresholdCorrectionFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = Do you want the output image to be binary (black/white) or grayscale?
%choiceVAR10 = Grayscale
%choiceVAR10 = Binary
%inputtypeVAR10 = popupmenu
BinaryOrGray = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

EdgedImage = OrigImage;

if isnan(Threshold)
    Threshold = [];
end

if isnan(ThresholdCorrectionFactor)
    ThresholdCorrectionFactor = 1;
end

Direction = lower(Direction);
Method = lower(Method);

if strcmpi(Method, 'roberts')
    if strcmpi(Thinning, 'thinning')
       Thinning = 'thinning';
    elseif strcmpi(Thinning, 'no thinning')
       Thinning = 'nothinning';
    end
    
    [EdgedImage, AutoThresh] = edge(OrigImage, Method, Threshold, Thinning);
    
    if ThresholdCorrectionFactor ~= 1
        if (ThresholdCorrectionFactor * AutoThresh > 1) ||  (ThresholdCorrectionFactor * AutoThresh < 0)
           CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Thinning);
    end
elseif strcmpi(Method, 'canny')
    if isnan(Sigma) || Sigma <= 0
        Sigma = 1; %canny's default is 1
        CPwarndlg(['The Sigma value you have entered in ', ModuleName, ' may be out of bounds. We set the value to 1']);
    end

    if isnan(CannyLowThreshold)
        CannyLowThreshold = [];
    end
    
    [EdgedImage, AutoThresh] = edge(OrigImage, Method, [CannyLowThreshold Threshold], Sigma);
    
    if ThresholdCorrectionFactor ~= 1
        if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
            CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Sigma);
    end
elseif  strcmpi(Method, 'log')
    if isnan(Sigma) || Sigma <= 0
        Sigma = 2; %log's default is two
        CPwarndlg(['The Sigma value you have entered in ', ModuleName, ' may be out of bounds. We set the value to 2']);
    end

    if isnan(CannyLowThreshold)
        CannyLowThreshold = [];
    end
    
    [EdgedImage, AutoThresh] = edge(OrigImage,Method, [CannyLowThreshold Threshold], Sigma);
    
    if ThresholdCorrectionFactor ~= 1
         if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
            CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Sigma);
    end
elseif strcmpi(Method, 'sobel') || strcmpi(Method, 'prewitt')
    [EdgedImage, AutoThresh] = edge(OrigImage, Method, Threshold, Direction);
    
    if ThresholdCorrectionFactor ~= 1
        if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
            CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Direction);
    end
end

if ~strcmp(BinaryOrGray,'Binary')
    EdgedImage = double(EdgedImage);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
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
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(EdgedImage,handles);
    title('Edged Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(OutputName) = EdgedImage;