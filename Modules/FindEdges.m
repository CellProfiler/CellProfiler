function handles = FindEdges(handles)

% Help for the Find Edges module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Identifies edges in an image, which can be used as the basis for object
% identification or other downstream image processing.
% *************************************************************************
%
% This module finds the edges of objects in a grayscale image, usually
% producing a binary (black and white) image where the edges are white and
% the background is black. The ratio method can optionally produce a
% grayscale image where the strongest edges are brighter and the smoothest
% parts of the image are darker. It works best when the objects of interest
% are black and the background is white.
%
% Settings:
%
% Threshold: Enter the desired threshold or have CellProfiler calculate one
% automatically. The methods use different processes to calculate the
% automatic threshold.
%
% Threshold Adjustment Factor: This value will be multiplied by the
% threshold (the numerical value you entered, or the automatically
% calculated one if desired) used for edge detection.
%
% Method: There are several methods that can be used to identify edges:
%   Ratio Method - This method first applies two smoothing filters to the
%                  image (sum of squares and square of sums), and then
%                  takes the ratio of the two resulting images to determine
%                  the edges. The filter size is then very important in
%                  this method. The larger the filter size, the thicker the
%                  edges will be. The recommended size is 8 pixels, or
%                  roughly half the width of the objects you wish to edge.
%                  This method is taken from CJ Cronin, JE Mendel, S
%                  Mukhtar, Y-M Kim, RC Stirbl, J Bruck and PW Sternberg,
%                  An automated system for measuring parameters of nematode
%                  sinusoidal movement, BMC Genetics, 6:5, 2005 available
%                  here: http://www.biomedcentral.com/1471-2156/6/5
%   Sobel Method - finds edges using the Sobel approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   Prewitt Method - finds edges using the Prewitt approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   Roberts Method - finds edges using the Roberts approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   LoG Method -   This method first applies a Laplacian of Gaussian filter
%                  to the image and then finds zero crossings.
%   Canny Method - The Canny method finds edges by looking for local maxima
%                  of the gradient of the image. The gradient is calculated
%                  using the derivative of a Gaussian filter. The method
%                  uses two thresholds, to detect strong and weak edges,
%                  and includes the weak edges in the output only if they
%                  are connected to strong edges. This method is therefore
%                  less likely than the others to be fooled by noise, and
%                  more likely to detect true weak edges.
%
% Size of smoothing filter (for Ratio method only): A square of NxN will be used for the filter, where N is the size you
% specify here. See method description above for further information.
%
% Binary or Grayscale (for Ratio method only): The image produced by this
% module can be grayscale (varying shaed of gray) or binary (black and
% white). The choice depends on what you intend to use the resulting image
% for.
%
% Edge Thinning (for Sobel and Roberts methods): If thinning is selected,
% edges found will be thinned out into a line (if possible). Specifying the
% 'nothinning' option can speed up the operation of the algorithm by
% skipping the additional edge thinning stage.
%
% Direction (for Sobel and Prewitt methods): It gives you the option of
% identifying all edges, or just those that are predominantly horizontal or
% vertical.
%
% Sigma (LoG and Canny): Standard deviation of the gaussian filter

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image in which you want to find edges?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the image with edges identified?
%defaultVAR02 = EdgedImage
%infotypeVAR02 = imagegroup indep
OutputName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter an absolute threshold in the range [0,1], or type '/' to calculate automatically.
%defaultVAR03 = /
OrigThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the Threshold Adjustment Factor, or leave it = 1.
%defaultVAR04 = 1
ThresholdCorrectionFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Choose an edge-finding method:
%choiceVAR05 = Ratio
%choiceVAR05 = Sobel
%choiceVAR05 = Prewitt
%choiceVAR05 = Roberts
%choiceVAR05 = LoG
%choiceVAR05 = Canny
Method = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = For RATIO method, enter the filter size (in pixels):
%defaultVAR06 = 8
SizeOfSmoothingFilter = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,06}));

%textVAR07 = For RATIO method, do you want the output image to be binary (black/white) or grayscale?
%choiceVAR07 = Binary
%choiceVAR07 = Grayscale
BinaryOrGray = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = For SOBEL and ROBERTS methods, do you want edge thinning?
%choiceVAR08 = Yes
%choiceVAR08 = No
Thinning = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = For SOBEL and PREWITT methods, which edges do you want to find?
%choiceVAR09 = All
%choiceVAR09 = Horizontal
%choiceVAR09 = Vertical
Direction = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = For LoG and CANNY methods, enter the value of sigma. Use '/' to calculate automatically.
%defaultVAR10 = /
Sigma = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = For CANNY method, enter the low threshold. Use '/' to calculate automatically.
%defaultVAR11 = /
CannyLowThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if isnan(ThresholdCorrectionFactor)
    ThresholdCorrectionFactor = 1;
end

if strcmpi(Thinning, 'Yes')
    Thinning = 'thinning';
elseif strcmpi(Thinning, 'No')
    Thinning = 'nothinning';
end

CalculateThreshold = 0;
if strcmp(OrigThreshold,'/')
    CalculateThreshold = 1;
else
    AutoThresh = str2double(OrigThreshold);
    if isnan(AutoThresh)
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad threshold entry']))
            CPwarndlg(['The threshold value you have entered in ', ModuleName, ' is out of bounds. It must be greater than 0 or ''/'' to use the default value. It has been changed to the default value.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad threshold entry'],'replace');
        end
        CalculateThreshold = 1;
    end
end
if strcmpi(Method, 'roberts')
    FourthVariable = Thinning;
elseif strcmpi(Method, 'canny')
    if strcmp(Sigma,'/')
        Sigma = 1; %canny's default is 1
    else
        Sigma = str2double(Sigma);
        if isnan(Sigma) || Sigma <= 0
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad Sigma entry']))
                CPwarndlg(['The Sigma value you have entered in ', ModuleName, ' is out of bounds. It must be greater than 0 or ''/'' to use the default value. Changing to the default value.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad Sigma entry'],'replace');
            end
            Sigma = 1;
        end
    end
    FourthVariable = Sigma;
    if strcmp(CannyLowThreshold,'/')
        CannyLowThreshold = [];
    else
        CannyLowThreshold = str2double(CannyLowThreshold);
        if isnan(CannyLowThreshold)
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad Canny low threshold entry']))
                CPwarndlg(['The low threshold value you have entered in ', ModuleName, ' is out of bounds. It must be greater than 0 or ''/'' to use the default value. Changing to the default value.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad Canny low threshold entry'],'replace');
            end
            CannyLowThreshold = [];
        end
    end
elseif  strcmpi(Method, 'log')
    if strcmp(Sigma,'/')
        Sigma = 2; %log's default is 1
    else
        Sigma = str2double(Sigma);
        if isnan(Sigma) || Sigma <= 0
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad LoG Sigma entry']))
                CPwarndlg(['The Sigma value you have entered in ', ModuleName, ' is out of bounds. It must be greater than 0 or ''/'' to use the default value. Changing to the default value.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad LoG Sigma entry'],'replace');
            end
            Sigma = 2;
        end
    end
    FourthVariable = Sigma;
elseif strcmpi(Method, 'sobel') || strcmpi(Method, 'prewitt')
    if strcmpi(Direction,'All')
        Direction = 'both';
    end
    FourthVariable = Direction;
end

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','DontCheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Note that we use the edge function to calculate automatic thresholds,
%%% because the methods use different threshold calculating functions.
%%% Even though it adds time to run edge twice, the help for the edge
%%% function indicates that this is the best way to calculate the automatic
%%% threshold.

%%% The first section is for the built in MATLAB functions, NOT the ratio
%%% method.
if ~strcmpi(Method,'ratio')

    %%% For the case where we want to automatically calculate a threshold.
    if CalculateThreshold
        %%% Note that we do not need to add the thinning variable to the
        %%% end of the edge function's variables for sobel because it does
        %%% not affect the calculation of the automatic threshold.
        [EdgedImage, AutoThresh] = edge(OrigImage, Method, [], FourthVariable); %#ok
    end

    if strcmpi(Method,'canny')
        AutoThresh = [CannyLowThreshold AutoThresh];
    end

    ThresholdUsed = ThresholdCorrectionFactor * AutoThresh;
    %%% Normally ThresholdUsed is a single numerical value, but we use max
    %%% and min because for the canny method the threshold has two values.
    if (max(ThresholdUsed) > 1) ||  (min(ThresholdUsed) < 0)
        CPwarndlg(['The Threshold Correction Factor you entered in ', ModuleName, ' resulted in a threshold greater than 1 or less than zero. You might need to adjust the Threshold Correction Factor so that when multiplied by the automatically calculated threshold (or the absolute threshold you entered), the value is still in the range 0 to 1.']);
    end

    if strcmpi(Method, 'sobel')
        EdgedImage = edge(OrigImage, Method, ThresholdUsed, FourthVariable, Thinning);
    else
        EdgedImage = edge(OrigImage, Method, ThresholdUsed, FourthVariable);
    end

elseif strcmpi(Method,'ratio')
    if (isnan(SizeOfSmoothingFilter) || (SizeOfSmoothingFilter < 1))
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad filter size entry']))
            CPwarndlg(['The smoothing filter size you entered in the ' ModuleName ' was invalid. Changing to the default value.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad filter size entry'],'replace');
        end
        SizeOfSmoothingFilter = 8;
    end
    Sq = CPsmooth(OrigImage,'Q',SizeOfSmoothingFilter,0);
    Mn = CPsmooth(OrigImage,'S',SizeOfSmoothingFilter,0);
    EdgedImage = Mn./Sq;

    if strcmp(BinaryOrGray,'Binary')
        %%% For the case where we want to automatically calculate a threshold.
        if CalculateThreshold
            [handles,AutoThresh] = CPthreshold(handles,'Otsu Global','01','0','1',1,EdgedImage,['Edged_',ImageName],ModuleName);
        else
            if AutoThresh > 1 || AutoThresh < 0
                CPwarndlg(['The threshold you entered in ', ModuleName,' should normally be between 0 and 1.']);
            end
        end
        if ThresholdCorrectionFactor ~= 1
            ThresholdUsed = ThresholdCorrectionFactor * AutoThresh;
            if (ThresholdUsed > 1) || (ThresholdUsed < 0)
                CPwarndlg(['The Threshold Correction Factor you entered in ', ModuleName, ' resulted in a threshold greater than 1 or less than zero. You might need to adjust the Threshold Correction Factor so that when multiplied by the automatically calculated threshold (or the absolute threshold you entered), the value is still in the range 0 to 1.']);
            end
        else
            ThresholdUsed = AutoThresh;
        end
        EdgedImage = im2bw(EdgedImage, ThresholdUsed);
    end

    if strcmp(Thinning,'thinning')
        EdgedImage = imerode(EdgedImage,strel('disk',3));
    end
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
    title(OutputName);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(OutputName) = EdgedImage;
if ~strcmpi(Method,'ratio') || ~strcmpi(BinaryOrGray,'Grayscale')
    if strcmpi(Method,'canny')
        handles = CPaddmeasurements(handles,'Image','OrigThreshold',['Edged_',ImageName],ThresholdUsed(1));
        handles = CPaddmeasurements(handles,'Image','OrigThreshold',['CannyLowEdged_',ImageName],ThresholdUsed(2));
    else
        if ~(strcmpi(Method,'ratio') && strcmp(BinaryOrGray,'Binary') && CalculateThreshold)
            handles = CPaddmeasurements(handles,'Image','OrigThreshold',['Edged_',ImageName],ThresholdUsed);
        end
    end
end
