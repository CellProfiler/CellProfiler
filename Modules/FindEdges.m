function handles = FindEdges(handles)

% Help for the Find Edges module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Employs several methods to identify edges in an image. Once the edges are
% found, a grayscale or binary image showing them will be produced.
% *************************************************************************
%
% This module will find the edges of objects in an image. You can choose
% amongst several methods of finding the edges. All of them, except the
% Ratio method, will return a binary image. The Ratio method can return a
% binary image or a grayscale image depending on your choice.
%
% Settings:
%
% Method: There are several different methods that can be used to identify
% edges. The user is able to pick between Roberts, Sobel, Prewitt, LoG,
% Canny, and Ratio. These methods all implement different techniques and
% algorithms which are suited towards different images.
%   Sobel Method - finds edges using the Sobel approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   Prewitt Method - finds edges using the Prewitt approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   Roberts Method - finds edges using the Roberts approximation to the
%                  derivative. It returns edges at those points where the
%                  gradient of the image is maximum.
%   LoG Method - This method first applies a Laplacian of Gaussian filter
%                to the image and then finds zero crossings.
%   Canny Method - The Canny method finds edges by looking for local maxima
%                  of the gradient of the image. The gradient is calculated
%                  using the derivative of a Gaussian filter. The method
%                  uses two thresholds, to detect strong and weak edges,
%                  and includes the weak edges in the output only if they
%                  are connected to strong edges. This method is therefore
%                  less likely than the others to be fooled by noise, and
%                  more likely to detect true weak edges.
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
%
% To find the best method for an image, you can compare the results of each
% by eye.
%
% Threshold: Enter the desired threshold or have CellProfiler choose one
% automatically. A different process is used to determine the threshold for
% each method.
% 
% Edge Thinning: This setting only applies to the Sobel, Prewitt, and
% Roberts methods. If thinning is selected, edges found will be thinned out
% into a line (if possible).
% 
% Direction: This setting applies only for the Sobel and Prewitt methods.
% It gives you the option of identifying either only horizontal or vertical
% edges or both.
%
% Threshold Factor: The Threshold factor specified by the user will be
% multiplied with the threshold used for edge detection.
%
% Size of smoothing filter: This setting applies only for the Ratio method.
% A square of NxN will be used for the filter, where N is the size you
% specify here. See method description above for further information.
%
% Binary or Grayscale: This setting applies only for the Ratio method. It
% allows the option of having the resulting image to have varying degrees
% of Gray (Grayscale) or of the resulting image being strictly black and
% white (Binary).

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
%defaultVAR02 = EdgedImage
%infotypeVAR02 = imagegroup indep
OutputName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What method would you like to use?
%choiceVAR03 = Sobel
%choiceVAR03 = Prewitt
%choiceVAR03 = Roberts
%choiceVAR03 = LoG
%choiceVAR03 = Canny
%choiceVAR05 = Ratio
%inputtypeVAR03 = popupmenu
Method = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want the threshold to be?  Put in a '/' to have one picked automatically. 
%defaultVAR04 = /
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = For the Sobel, Prewitt, and Roberts methods, do you want edge thinning?
%choiceVAR05 = Thinning
%choiceVAR05 = No Thinning
%inputtypeVAR05 = popupmenu
Thinning = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For the Sobel and Prewitt methods, which direction do you want to use?
%choiceVAR06 = Both
%choiceVAR06 = Horizontal
%choiceVAR06 = Vertical
%inputtypeVAR06 = popupmenu
Direction = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = For the LoG and Canny methods, what is the value of sigma? Use '/' for defaults.
%defaultVAR07 = /
Sigma = str2double(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For the Canny method, what is the low threshold? Use '/' for defaults.
%defaultVAR08 = /
CannyLowThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = What is the Threshold Correction Factor? The Threshold can be automatically found, then multiplied by this factor.
%defaultVAR09 = 1
ThresholdCorrectionFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = For the Ratio method, specify the filter size (in pixels) that you would like to use. (see help for details)
%defaultVAR10 = 8
SizeOfSmoothingFilter = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = For the Ratio method, do you want the output image to be binary (black/white) or grayscale?
%choiceVAR11 = Grayscale
%choiceVAR11 = Binary
%inputtypeVAR11 = popupmenu
BinaryOrGray = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Variable check
if isnan(Threshold)
    Threshold = [];
end

if isnan(ThresholdCorrectionFactor)
    ThresholdCorrectionFactor = 1;
end

if strcmpi(Thinning, 'thinning')
    Thinning = 'thinning';
elseif strcmpi(Thinning, 'no thinning')
    Thinning = 'nothinning';
end

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

EdgedImage = OrigImage;

if strcmpi(Method, 'roberts')
    
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

    [EdgedImage, AutoThresh] = edge(OrigImage,Method, Threshold, Sigma);
    
    if ThresholdCorrectionFactor ~= 1
        if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
            CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Sigma);
    end

elseif strcmpi(Method, 'sobel') || strcmpi(Method, 'prewitt')
    [EdgedImage, AutoThresh] = edge(OrigImage, Method, Threshold, Direction, Thinning);
    
    if ThresholdCorrectionFactor ~= 1
        if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
            CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
        end
        EdgedImage = edge(OrigImage, Method, AutoThresh * ThresholdCorrectionFactor, Direction);
    end

elseif strcmpi(Method,'ratio')
    if isnan(SizeOfSmoothingFilter)
        error(['Image processing was canceled in the ' ModuleName ' module because the size of smoothing filter you specified was invalid.']);
    else
        SizeOfSmoothingFilter = min(30,max(1,floor(SizeOfSmoothingFilter)));
    end
    Sq = CPsmooth(OrigImage,'Q',SizeOfSmoothingFilter,0);
    Mn = CPsmooth(OrigImage,'S',SizeOfSmoothingFilter,0);
    EdgedImage = Mn./Sq;
    %%% The ratio image has really weird numbers, put it in the 0-1 range:
    [handles, EdgedImage] = CPrescale(handles,EdgedImage,'S',[]);
    if strcmp(BinaryOrGray,'Binary')
        if isempty(Threshold)
            AutoThresh = graythresh(EdgedImage);
            if ThresholdCorrectionFactor ~= 1
                if (ThresholdCorrectionFactor * AutoThresh > 1) || (ThresholdCorrectionFactor * AutoThresh < 0)
                    CPwarndlg(['The Threshold Correction Factor you have entered in ', ModuleName, ' may be out of bounds.']);
                end
                Threshold = ThresholdCorrectionFactor * AutoThresh;
            else
                Threshold = AutoThresh;
            end
        end
        EdgedImage = im2bw(EdgedImage, Threshold);
    end
%     if strcmp(Thinning,'thinning')
%         EdgedImage = imerode(EdgedImage,strel('disk',3));
%     end
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
