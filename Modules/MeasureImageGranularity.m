function handles = MeasureImageGranularity(handles,varargin)

% Help for the Measure Image Granularity module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module measures the image granularity as described by Ilya Ravkin.
% *************************************************************************
%
% Image granularity can be useful to measure particular assays, in
% particular the "Transfluor" assay which depends on cellular
% texture/smoothness.
%
% Features measured:      Feature Number:
% GS1                 |        1
% GS2                 |        2
% GS3                 |        3
% GS4                 |        4
% GS5                 |        5
% GS6                 |        6
% GS7                 |        7
% GS8                 |        8
% GS9                 |        9
% GS10                |        10
% GS11                |        11
% GS12                |        12
% GS13                |        13
% GS14                |        14
% GS15                |        15
% GS16                |        16
%
% Settings for this module:
%
% Subsampling size: Only a subsample of the image is processed, to speed up
% the calculation. Increasing the fraction will increase the accuracy but
% will require more processing time. Images are typically of higher
% resolution than is required for this step, so the default is to subsample
% 25% of the image. For low-resolution images, increase the subsampling
% fraction and for high-resolution images, decrease the subsampling
% fraction. More detail: The subsampling fraction is to compensate for the
% usually oversampled images; at least they are oversampled in both
% Transfluor library image sets. See
% http://www.ravkin.net/presentations/Statistical%20properties%20of%20algor
% ith ms%20for%20analysis%20of%20cell%20images.pdf slides 27-31, 49-50.
% Subsampling by 1/4 reduces computation time by (1/4)^3 because the size
% of the image is (1/4)^2 of original and length of granular spectrum can
% be 1/4 of original. Moreover, the results were actually a little better
% with subsampling, which is probably because with subsampling the
% individual granular spectrum components can be used as features, whereas
% without subsampling a feature should be a sum of several adjacent
% granular spectrum components. The recommendation on the numerical value
% can't be given in advance; an analysis like in the above reference is
% required before running the whole set.
%
% Subsample fraction: Background removal is just to remove low frequency in
% the image. Any method can be used. We subtract a highly open image. To do
% it fast we subsample the image first. The subsampling fraction is usually
% [0.125 - 0.25].  This is highly empirical. The significance of background
% removal in the context of granulometry is only in that image volume at
% certain thickness is normalized by total volume, which depends on how the
% background was removed.
%
% Structuring element size: Radius of the structuring element (in
% subsampled image). Radius of structuring element after subsampling is
% usually [6-16]. It is better to think of this radius in the original
% image scale and then to multiply by subsampling fraction. In the original
% image scale it should be [30-60]. This is highly empirical.
%
% Granular Spectrum Length (default = 16): Needs a trial run to see which
% Granular Spectrum Length yields informative measurements.
%
%
% References for Granular Spectrum:
% J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic
% Press, London, 1989 Maragos,P. "Pattern spectrum and multiscale shape
% representation", IEEE Transactions on Pattern Analysis and Machine
% Intelligence, 11, N 7, pp. 701-716, 1989
%
% L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae,
% 41, No. 1-2, pp. 57-90, IOS Press, 2000.
%
% L.Vincent "Morphological Area Opening and Closing for Grayscale Images",
% Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp.
% 197-208, 1992.
%
% I.Ravkin, V.Temov "Bit representation techniques and image processing",
% Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow,
% 1988 (in Russian)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order: Unchanged from current variables/settings in
% MATLAB CP)
%
% Anne recommends changing fractions to percentages, if that seems
% reasonable to you.
%
% Sounds also like we should extract out the background removal step from
% this module, since this can be done in a more controlled way with
% CorrectIllumCalculate. But I'm not sure the impact that would have on the
% algorithm. David can help test any alterations we make with his SBS test
% images, where the Z' factor has been calculated using this method of
% measuring granularity.  The help for this module is from Ilya Ravkin and
% could use more explanation.

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image whose granularity you would like to measure?
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

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || strcmp(varargin{2},'Image')
                result = { 'Granularity' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Granularity') &&...
                strcmp(varargin{2},'Image')
                result = arrayfun(...
                    @(x) num2str(x),...
                    1:GranularSpectrumLength,'UniformOutput',false);
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%
%%% ANALYSIS %%%
%%%%%%%%%%%%%%%%

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%ANALYZE
B = imresize(OrigImage, SubSampleSize, 'bilinear'); %RESULTS ON iCyte IMAGES WITH THIS SUBSAMPLING ARE AS GOOD OR BETTER THAN WITH ORIGINALS
C = backgroundremoval(B, ImageSampleSize, ElementSize);
gs = granspectr(C, GranularSpectrumLength);

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
    hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber);
    CPimagesc(OrigImage,handles,hAx);
    title(hAx,['Input Images, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    hAx=subplot(2,1,2,'Parent',ThisModuleFigureNumber);
    CPimagesc(C,handles,hAx);
    title(hAx,'Background Subtracted Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:GranularSpectrumLength
    handles = CPaddmeasurements ...
	      (handles, 'Image', ...
	       CPjoinstrings('Granularity', num2str(i), ImageName), gs(:,i));
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