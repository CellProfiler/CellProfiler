function handles = MeasureImageGranularity(handles)

% Help for the Measure Image Granularity module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module measures the image granularity as described by Ilya Ravkin.
% *************************************************************************
%
% Subsample Size:
% Subsampling of the image for background removal, given as fraction
%
% Structuring Element Size:
% Radius of structuring element (in subsampled image)
% 
% References for Granular Spectrum:
% J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic
% Press, London, 1989 Maragos,P. “Pattern spectrum and multiscale shape
% representation”, IEEE Transactions on Pattern Analysis and Machine
% Intelligence, 11, N 7, pp. 701-716, 1989
%
% L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae,
% 41, No. 1-2, pp. 57-90, IOS Press, 2000.
%
% L.Vincent "Morphological Area Opening and Closing for Grayscale Images",
% Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp.
% 197-208, 1992.
%
% I.Ravkin, V.Temov “Bit representation techniques and image processing”,
% Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow,
% 1988 (in Russian)

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
% $Revision: 1843 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which image would you like to measure?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want the image subsample size to be?
%defaultVAR02 = 0.25
SubSampleSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = What fraction of the resulting image do you want to sample?
%defaultVAR03 = 0.25
ImageSampleSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = What is the size of the structuring element?
%defaultVAR04 = 10
ElementSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = What do you want to be the length of the granular spectrum?
%defaultVAR05 = 16
GranularSpectrumLength = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%
%%% ANALYSIS %%%
%%%%%%%%%%%%%%%%

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%ANALYZE
B = imresize(OrigImage, SubSampleSize, 'bilinear'); %RESULTS ON iCyte IMAGES WITH THIS SUBSAMPLING ARE AS GOOD OR BETTER THAN WITH ORIGINALS
C = backgroundremoval(B, ImageSampleSize, ElementSize);
gs = granspectr(C, GranularSpectrumLength);

handles.Measurements.Image.([ImageName 'Spectrum']){handles.Current.SetBeingAnalyzed}=gs;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
        %%% Add extra space for the text at the bottom.
        Position = get(ThisModuleFigureNumber,'position');
        set(ThisModuleFigureNumber,'position',[Position(1),Position(2)-40,Position(3),Position(4)+40])
    end
    %%% A subplot of the figure window is set to display the original
    %%% image.
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Images, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(C,handles);
    title('Background Subtracted Image');
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function br=backgroundremoval(img,bgrsub,bgrthick)
%REMOVE BACKGROUND BY SUBTRACTING OPEN IMAGE, LIKE TOP HAT, BUT FOR SPEED - SUBSAMPLE
%PARAMETERS:
% img - THE IMAGE, MUST BE TWO-DIMENSIONAL, GRAYSCALE.
% bgrsub - SUBSAMPLING OF THE IMAGE FOR BACKGROUND REMOVAL GIVEN AS FRACTION
% bgrthick - RADIUS OF STRUCTURING ELEMENT (IN SUBSAMPLED IMAGE)
%EXAMPLE: br=backgroundremoval(img,0.25,10)

imr = imresize(img, bgrsub); %RESIZE DOWN
imo = imopen(imr,strel('disk',bgrthick)); %MAKE BACKGROUND IMAGE
imb = imresize(imo, size(img),'bilinear'); %RESIZE UP
br = imsubtract(img,imb); %SUBTRACT BACKGROUND IMAGE FROM THE ORIGINAL

function gs=granspectr(img,ng)
%CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
%GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
%J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
%Maragos,P. “Pattern spectrum and multiscale shape representation”, IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
%L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
%L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
%I.Ravkin, V.Temov “Bit representation techniques and image processing”, Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)

%THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
%BACKGROUND SHOULD BE REMOVED BEFORE THE CALCULATION OF THE GRANULAR SPECTRUM

%PARAMETERS:
% img - THE IMAGE, MUST BE TWO-DIMENSIONAL, GRAYSCALE
% ng - LENGTH OF GRANULAR SPECTRUM (NUMBER OF FEATURES GS01, GS02, ...)

gs = zeros(1, ng);
startmean = mean2(img);
ero = img;
currentmean = startmean;
for i = 1 : ng
    prevmean = currentmean;
    ero = imerode(ero,strel('diamond',1));
    rec = imreconstruct(ero,img,4);
    currentmean = mean2(rec);
    gs(i) = prevmean - currentmean;
end
gs = gs .* (100 / startmean);