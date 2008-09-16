function handles = DICTransform(handles)

% Help for the DIC Transform module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Transforms a DIC image to bas relief so that standard segmentation algorithms
% will function.
% *************************************************************************
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
% $Revision: 5025$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be transformed?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the transformed image?
%defaultVAR02 = TransformDIC
%infotypeVAR02 = imagegroup indep
TransformedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Choose a shear angle:
%choiceVAR03 = Main Diagonal
%choiceVAR03 = Anti Diagonal
%choiceVAR03 = Vertical
%choiceVAR03 = Horizontal
Shear = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Choose a transformation method:
%choiceVAR04 = Line Integration
%choiceVAR04 = Variance Filter
%choiceVAR04 = Hilbert Transform
%choiceVAR04 = Minimize Energy
Method = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = For LINE INTEGRATION method, enter the decay rate:
%defaultVAR05 = .9
DecayRate = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = For LINE INTEGRATION method, enter the perpindicular averaging decay rate:
%defaultVAR06 = .6
PerpDecayRate = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = For VARIANCE FILTER method, enter the standard deviation of the weighting function (in pixels):
%defaultVAR07 = 2.0
FilterStdev = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = For VARIANCE FILTER method, enter the filter size (in pixels):
%defaultVAR08 = 51
FilterSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = For HILBERT TRANSFORM method, enter the exponent for the suppression of higher frequencies:
%defaultVAR09 = 2.0
FreqSuppress = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = For HILBERT TRANSFORM method, enter alpha (effects frequency suppression and convergence rate):
%defaultVAR10 = 1.0
Alpha = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = For HILBERT TRANSFORM method, enter the number of iterations:
%defaultVAR11 = 20
HTIterations = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,11}));

%textVAR12 = For MINIMIZE ENERGY method, choose the error function:
%choiceVAR12 = Square
%choiceVAR12 = Absolute Value
%choiceVAR12 = Approximate Absolute Value
%choiceVAR12 = Cosine
ErrorFunction = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = For MINIMIZE ENERGY method, enter the weight given to smoothness:
%defaultVAR13 = .1
SmoothFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%textVAR14 = For MINIMIZE ENERGY method, enter the weight given to flatness:
%defaultVAR14 = .01
FlatFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,14}));

%textVAR15 = For MINIMIZE ENERGY method, enter the number of iterations:
%defaultVAR15 = 400
MEIterations = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,15}));

%textVAR16 = For MINIMIZE ENERGY method with APPROXIMATE ABSOLUTE VALUE error, enter the exponent (higher more closely approximates absolute value):
%defaultVAR16 = 400
Abs2Exp = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,16}));

%textVAR17 = For MINIMIZE ENERGY method with SIN error, enter the range:
%defaultVAR17 = 1.5
SinRange = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,17}));

%%%VariableRevisionNumber = 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','DontCheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Rotate accordingly.
if strcmpi(Shear, 'Anti Diagonal') || strcmpi(Shear, 'Horizontal')
    AlignedImage = imrotate(OrigImage,90);
else
    AlignedImage = OrigImage;
end

%%% Choose direction.
if strcmpi(Shear, 'Main Diagonal') || strcmpi(Shear, 'Anti Diagonal')
    Direction = 'diagonal';
else
    Direction = 'vertical';
end

%%% Transforms the image.
if strcmpi(Method, 'Line Integration')
	TransformedImage = CPlineintegration(AlignedImage, DecayRate, PerpDecayRate, Direction);
elseif strcmpi(Method, 'Variance Filter')
	TransformedImage = CPvariancefilter(AlignedImage, FilterSize, FilterStdev);
elseif strcmpi(Method, 'Hilbert Transform')
	TransformedImage = CPhilberttransform(AlignedImage, HTIterations, Alpha, FreqSuppress, Direction);
elseif strcmpi(Method, 'Minimize Energy')
    if strcmpi(ErrorFunction, 'Square')
    	TransformedImage = CPminimizeenergy(AlignedImage, SmoothFactor, FlatFactor, MEIterations, Direction);
    elseif strcmpi(ErrorFunction, 'Absolute Value')
        SmoothFactor = SmoothFactor * 200;
        FlatFactor = FlatFactor * 20;
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, 4e-4, 'abs', Direction);
    elseif strcmpi(ErrorFunction, 'Approximate Absolute Value')
        SmoothFactor = SmoothFactor * 200;
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, 1e-3, 'abs2', Direction, Abs2Exp);
    elseif strcmpi(ErrorFunction, 'Cosine')
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, .05, 'sin', Direction, SinRange);
    end
end

if strcmpi(Shear, 'Anti Diagonal') || strcmpi(Shear, 'Horizontal')
    TransformedImage = imrotate(TransformedImage,-90);
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
    %%% A subplot of the figure window is set to display the Transformed
    %%% Image.
    subplot(2,1,2); 
    CPimagesc(TransformedImage,handles); 
    title('Transformed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the transformed image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(TransformedImageName) = TransformedImage;
