function handles = FindEdges(handles)

% Help for the Find Edges module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% The FindEdges module employs serveral methods to identify any edges in an
% image.  Once the edges are found, a resulting binary image will be produced
% where only the edges are shown in white while everything else will be black. 
%
% Settings:
%   Method- There are several different methods that can be used to identify
% edges.  In this module, the user is able to picke between Roberts, Sobel,
% Prewitt, Log, and Canny.  These methods all implement different
% techniques and algorithms which are suited towards different images.
%       Sobel Method - finds edges using the Sobel approximation to the
%                      derivative. It returns edges at those points where 
%                      the gradient of which I is maximum. 
%       Prewitt Method - finds edges using the Prewitt approximation to the
%                      derivative. It returns edges at those points where
%                      the gradient of I is maximum. 
%       Roberts Method - finds edges using the Roberts approximation to the
%                      derivative. It returns edges at those points where
%                      the gradient of I is maximum. 
%       Canny - The Canny method finds edges by looking for local maxima of
%                      of the gradient of I. The gradient is calculated
%                      using the derivative of a Gaussian filter. The
%                      method uses two thresholds, to detect strong and
%                      weak edges, and includes the weak edges in the
%                      output only if they are connected to strong edges.
%                      This method is therefore less likely than the others
%                      to be fooled by noise, and more likely to detect
%                      true weak edges.
% To find the best method for an image, you can compare the results of each
% by eye.
%
%   Threshold - Enter the desired threshold or have CellProfiler choose one
% automatically.  A different process is used to determine the threshold for
% each method.
%   Edge Thinning - This setting only applies for the Roberts method.  
%   Direction - This setting applies only for Sobel and Prewitt methods.  It
% gives you the option of identifying either only horizontal or vertical
% edges or both.     
%   Threshold Factor- The Threshold factor specified by the user will be
% multiplied with the threshold used for edge detection.
%   Binary or GrayScale- This option allows the option of having the
% resulting image to have varying degrees of Gray (Grayscale) or of the
% resulting image being strictly black and white (Binary).
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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
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
