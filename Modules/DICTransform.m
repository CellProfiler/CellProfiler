function handles = DICTransform(handles)

% Help for the DIC Transform module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Transforms a DIC image to more readily enable object identification.
% 
% ************************************************************************
%
% Typically, identifying objects with algorithms like those in
% CellProfiler's "Identify" modules does not work well for DIC images
% because the interior of each object has similar intensity values as the
% background. This module provides several algorithms for transforming a
% DIC image to enhance the brightness of objects relative to the
% background.
% 
%
% Settings:
%
% * Transformation methods: Several options are provided. For most DIC
% images, the line integration and the energy minimization algorithms
% perform the best. For objects that are heavily textured (e.g., mice
% embryos), a simple normally weighted variance filter is usually more
% helpful.
%
% * Shear axis: Direction along which the "shadow" of the objects appears
% to lie.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Broad Institute of MIT and Harvard.
% Copyright 2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision: 5025$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

% pyCP notes:
% (1) There needs to be more Help on all the individual 
%   transformation method settings. In particular, there's no mention in
%   the help of when Hilbert transform is useful - can
% we provide some guidance on that?  More info might be found here:
%   http://wwwdev.broad.mit.edu/imaging/privatewiki/index.php/2008_06_09_DIC_Transformation_Module_(Imaging_Platform)
% (2) All of the settings below the transformation method
%   should be displayed only with the respective method.
% (3) The transformation method should be setting #3, because it's the
% biggest decision (aside from naming images).

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be transformed?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the transformed image?
%defaultVAR02 = TransformedDIC
%infotypeVAR02 = imagegroup indep
TransformedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Choose a shear axis:
%choiceVAR03 = Main diagonal
%choiceVAR03 = Antidiagonal
%choiceVAR03 = Vertical
%choiceVAR03 = Horizontal
Shear = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Choose a transformation method:
%choiceVAR04 = Line integration
%choiceVAR04 = Variance filter
%choiceVAR04 = Hilbert transform
%choiceVAR04 = Energy minimization
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

%textVAR12 = For ENERGY MINIMIZATION method, choose the error function:
%choiceVAR12 = Square
%choiceVAR12 = Absolute value
%choiceVAR12 = Approximate absolute value
%choiceVAR12 = Cosine
ErrorFunction = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = For ENERGY MINIMIZATION method, enter the weight given to smoothness:
%defaultVAR13 = .1
SmoothFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%textVAR14 = For ENERGY MINIMIZATION method, enter the weight given to flatness:
%defaultVAR14 = .01
FlatFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,14}));

%textVAR15 = For ENERGY MINIMIZATION method, enter the number of iterations:
%defaultVAR15 = 400
MEIterations = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,15}));

%textVAR16 = For ENERGY MINIMIZATION method with APPROXIMATE ABSOLUTE VALUE error, enter the exponent (higher more closely approximates absolute value):
%defaultVAR16 = 400
Abs2Exp = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,16}));

%textVAR17 = For ENERGY MINIMIZATION method with SIN error, enter the range:
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
if strcmpi(Shear, 'Antidiagonal') || strcmpi(Shear, 'Horizontal')
    AlignedImage = imrotate(OrigImage,90);
else
    AlignedImage = OrigImage;
end

%%% Choose direction.
if strcmpi(Shear, 'Main Diagonal') || strcmpi(Shear, 'Antidiagonal')
    Direction = 'diagonal';
else
    Direction = 'vertical';
end

%%% Transforms the image.
if strcmpi(Method, 'Line integration')
	TransformedImage = CPlineintegration(AlignedImage, DecayRate, PerpDecayRate, Direction);
elseif strcmpi(Method, 'Variance filter')
	TransformedImage = CPvariancefilter(AlignedImage, FilterSize, FilterStdev);
elseif strcmpi(Method, 'Hilbert transform')
	TransformedImage = CPhilberttransform(AlignedImage, HTIterations, Alpha, FreqSuppress, Direction);
elseif strcmpi(Method, 'Energy minimization')
    if strcmpi(ErrorFunction, 'Square')
    	TransformedImage = CPminimizeenergy(AlignedImage, SmoothFactor, FlatFactor, MEIterations, Direction);
    elseif strcmpi(ErrorFunction, 'Absolute value')
        SmoothFactor = SmoothFactor * 200;
        FlatFactor = FlatFactor * 20;
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, 4e-4, 'abs', Direction);
    elseif strcmpi(ErrorFunction, 'Approximate absolute value')
        SmoothFactor = SmoothFactor * 200;
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, 1e-3, 'abs2', Direction, Abs2Exp);
    elseif strcmpi(ErrorFunction, 'Cosine')
        TransformedImage = CPminimizeenergy2(AlignedImage, SmoothFactor, FlatFactor, MEIterations, .05, 'sin', Direction, SinRange);
    end
end

%%% Anti-rotate if rotated before.
if strcmpi(Shear, 'Antidiagonal') || strcmpi(Shear, 'Horizontal')
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
    hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber); 
    CPimagesc(OrigImage,handles,hAx); 
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Transformed
    %%% Image.
    hAx=subplot(2,1,2,'Parent',ThisModuleFigureNumber); 
    CPimagesc(TransformedImage,handles,hAx); 
    title(hAx,'Transformed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the transformed image to the
%%% handles structure so it can be used by subsequent modules.
handles = CPaddimages(handles,TransformedImageName,TransformedImage);
