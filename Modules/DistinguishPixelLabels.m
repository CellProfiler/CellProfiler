function handles = DistinguishPixelLabels(handles)

% Help for the Distinguish Pixel Labels module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Labels pixels as cell, nucleus, or background based on actin and DNA stain
% intensities and neighboring labels.
% *************************************************************************
%
% By using the belief propagation algorithm, this module finds a more accurate 
% label for each pixel in the field of view.  Use this module before the 
% IdentifyPrimAutomatic and IdentifySecondary modules, and instead of selecting 
% an automatic threshold there, enter the name of a binary image produced in 
% this module.
%
% Overview:
%   The goal of this module is to find the most accurate binary images that specify
% where in the image cells and nuclei exist.  Typically this is done by applying 
% a threshold to input grayscale images, where every pixel above the threshold 
% value is considered foreground and every pixel below, background.  
% Using ordinary thresholds, however, ignores a great deal of information; the 
% belief propagation algorithm improves upon this by labeling pixels as foreground
% or background based not only on their pixel intensity value but also on the 
% values of nearby pixels and a predetermined set of probabilities that neighboring 
% pixels of each type share the same label.
%   The belief propagation (BP) algorithm is used to solve inference problems--in 
% this case, to predict the best label (cell, nucleus, or background) for a set
% of pixels.  Treating the field of view as a pairwise markov random field, in
% which the observed pixel intensities from two stained images are the observable
% nodes and their corresponding labels are the hidden nodes, BP finds the 
% marginal probabilities (or "beliefs") for each label at each pixel.  See 
% Yedidia et. al. or scroll down to "Technical Description" for more 
% information.
%
% Yedidia, J. S., Freeman, W. T., and Weiss, Y. (2002).  Understanding
% belief propagation and its generalizations. Technical report, Mitsubishi
% Electric Research Labs., TR-2001-22.
%
% Settings:
% Peak pixel intensity selection method:
% This module depends finding three "peaks" that represent the DNA and actin
% staining intensities which are most common among each label.  You can choose
% to have these peaks automatically calculated for each image in the set, or 
% you can have a histogram displayed for manual selection.  For either manual 
% option, the peaks remain the same for every image in the set.  Per-image 
% calculation is more reliable unless all images seem to have about the same 
% overall brightness and area covered by cells and nuclei.  
%%%% SHOULD THIS OPTION BE ERASED? WE NEVER USE ANYTHING BUT AUTO-PERIMAGE 
%%%% PEAK CALCULATION!
%
%
% The actin scaling factor input variable will be used to scale the
% distances between the actin peak and the real pixels by 1/input.  This
% will be inverted again for the probabilities; thus, a higher number makes
% the probabilities of actin foreground and actin background more powerful
% relative to the probabilities from the DNA staining.
%
% With Testing Mode turned on, there will be 4 files saved to the
% handles.Pipeline structure that are the beliefs that would be calculated
% after each directional message passing during the final propagation step.
% These are saved in handles.Pipeline.BeliefsAfterPass1 (and 2-4), and can
% be accessed by other modules by selecting "Other..." and typing in
% BeliefsAfterPass1 (or 2-4).  Otherwise, beliefs are only calculated once
% (meaning that the module will run faster).
%
% See also IdentifyPrimAutomatic, IdentifySecondary

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
% $Revision: 1750 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images of nuclei?
%infotypeVAR01 = imagegroup
NucleiImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the images of cells?
%infotypeVAR02 = imagegroup
CellsImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the binary image representing area labeled as nuclei?
%defaultVAR03 = BPNuclei
NucleiOutputName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%infotypeVAR03 = imagegroup indep

%textVAR04 = What do you want to call the binary image representing area labeled as cells?
%defaultVAR04 = BPCells
CellsOutputName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%infotypeVAR04 = imagegroup indep

%textVAR05 = What do you want to call the binary image representing area labeled as background?
%defaultVAR05 = Do not save
BackgroundOutputName = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%infotypeVAR05 = imagegroup indep

%textVAR06 = What do you want to call the grayscale image showing all three binaries?
%defaultVAR06 = Do not save
GrayscaleOutputName = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%infotypeVAR06 = imagegroup indep

%textVAR07 = Choose peak pixel intensity selection method:
%choiceVAR07 = Automatic - Per Image
%choiceVAR07 = Automatic - Per Set
%choiceVAR07 = Numeric - Per Set
PeakSelectionMethod = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = How many times do you want to propagate messages in each direction?
%defaultVAR08 = 5
NumberOfPropagationsString = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to use as a sigma value for gaussian probability calculation in the phi subfunction?
%defaultVAR09 = 0.5
SigmaValueString = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Enter a positive scaling factor for enhancing the actin messages in the phi subfunction.
%defaultVAR10 = 2
ActinScalingFactorString = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What did you call the illumination correction matrix for nuclei? (optional - leave as "none" if you do not wish to correct illumination) 
%infotypeVAR11 = imagegroup
%choiceVAR11 = none
LoadedNIllumCorrName = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu custom

%textVAR12 = What did you call the illumination correction matrix for nuclei? (optional - leave as "none" if you do not wish to correct illumination)
%infotypeVAR12 = imagegroup
%choiceVAR12 = none
LoadedCIllumCorrName = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 = For NUMERIC, enter the coordinates for the peak in nucleus, cell, and background pixel intensity values as (NucleiX,NucleiY,CellsX,CellsY,BGX,BGY).
%defaultVAR13 = 100,100,150,150,200,200
AllPeakInputValuesString = char(handles.Settings.VariableValues{CurrentModuleNum,13}); 

%textVAR14 = Do you want to run in test mode where beliefs are calculated after each direction of message passing during the final propagation? (see help for details)
%choiceVAR14 = No
%choiceVAR14 = Yes
TestingMode = char(handles.Settings.VariableValues{CurrentModuleNum,14});
%inputtypeVAR14 = popupmenu

%textVAR15 = What version of the phi subfunction do you want to use? "Hacky" doesn't work with numeric peak selection. (REMOVE THIS OPTION OR RENAME IT AFTER DEVELOPMENT IS COMPLETE!)
%choiceVAR15 = Normal
%choiceVAR15 = Hacky
PhiVersion = char(handles.Settings.VariableValues{CurrentModuleNum,15});
%inputtypeVAR15 = popupmenu

%%%VariableRevisionNumber = 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Parses user input for the number of propagations and the sigma value
NumberOfProps = str2num(NumberOfPropagationsString); %#ok Ignore MLint
handles.Pipeline.SigmaValue = str2num(SigmaValueString); %#ok
handles.Pipeline.ActinScalingFactor = str2num(ActinScalingFactorString); %#ok

%%% If this is the first image set (and only then), displays or
%%% creates the 2-Dimensional histogram to interactively find the peak
%%% illumination values for cell, nucleus, and background OR loads the
%%% numeric, preset peak illumination values
if handles.Current.SetBeingAnalyzed == 1

    if strncmp(PeakSelectionMethod,'Numeric',7)

        %%% Parses user input for the three points
        AllPixelInputs = str2num(AllPeakInputValuesString); %#ok ignore MLint
        handles.Pipeline.CellsPeak = [AllPixelInputs(3);AllPixelInputs(4)];
        handles.Pipeline.NucleiPeak = [AllPixelInputs(1);AllPixelInputs(2)];
        handles.Pipeline.BackgroundPeak = [AllPixelInputs(5);AllPixelInputs(6)];

    elseif strcmp(PeakSelectionMethod,'Automatic - Per Set')

        %%% Retrieves the path where the images of each type are stored from
        %%% the handles structure, looking in fields set by LoadImages
        fieldname = ['Pathname', NucleiImageName];
        try NucleiPathname = handles.Pipeline.(fieldname);
        catch error(['Image processing was canceled in the ', ModuleName, ' module because all the images must exist prior to processing the first cycle through the pipeline.  This means that the ',ModuleName, ' module must be run immediately after a Load Images module.'])
        end
        fieldname = ['Pathname', CellsImageName];
        try CellsPathname = handles.Pipeline.(fieldname);
        catch error(['Image processing was canceled in the ', ModuleName, ' module because all the images must exist prior to processing the first cycle through the pipeline.  This means that the ',ModuleName, ' module must be run immediately after a Load Images module.'])
        end
        %%% Retrieves the lists of all cell and nuclei image filenames
        %%% by looking in the handles structure to fields set by the
        %%% LoadImages module
        fieldname = ['FileList', NucleiImageName];
        NucleiFileList = handles.Pipeline.(fieldname);
        fieldname = ['FileList', CellsImageName];
        CellsFileList = handles.Pipeline.(fieldname);

        handle1 = CPhelpdlg(['Preliminary calculations are under way for the ', ModuleName, ' module.  Subsequent cycles skip this step and will run much more quickly.  Initial calculations are complete when this window closes.']);

        %%% initialize image log stores
        [PeekedAtNucleiImage,handles] = CPimread(fullfile(NucleiPathname,char(NucleiFileList(1))),handles);
        AllNucleiImages = zeros(min(20,length(NucleiFileList)),numel(PeekedAtNucleiImage));
        AllCellsImages = zeros(min(20,length(NucleiFileList)),numel(PeekedAtNucleiImage));

        %%% Loops through all loaded images during this first cycle (if
        %%% there are fewer than 20)
        if length(NucleiFileList) <= 20
            for i=1:length(NucleiFileList)
                [LoadedNucleiImage, handles] = CPimread(fullfile(NucleiPathname,char(NucleiFileList(i))),handles);
                [LoadedCellsImage, handles] = CPimread(fullfile(CellsPathname,char(CellsFileList(i))),handles);
                AllNucleiImages(i,:) = imageLog(LoadedNucleiImage(:));
                AllCellsImages(i,:) = imageLog(LoadedCellsImage(:));
            end
        else
            counter = 0;
            factor = floor(length(NucleiFileList) / 20);
            while counter < 20
                [LoadedNucleiImage, handles] = CPimread(fullfile(NucleiPathname,char(NucleiFileList(counter*factor))),handles);
                [LoadedCellsImage, handles] = CPimread(fullfile(CellsPathname,char(CellsFileList(counter*factor))),handles);
                AllNucleiImages(counter,:) = imageLog(LoadedNucleiImage(:));
                AllCellsImages(counter,:) = imageLog(LoadedCellsImage(:));
            end
        end

        %%% Defaults to Otsu's method, ignores potential mask
        NucThreshold = graythresh(AllNucleiImages);
        CellThreshold = graythresh(AllCellsImages);

        %%% Gets the means in each section of the image that we believe to have
        %%% a certain pixel label
        NucleiBGCellBGMean = 256*mean(AllNucleiImages(AllNucleiImages <= NucThreshold & AllCellsImages <= CellThreshold));
        CellsBGNucBGMean = 256*mean(AllCellsImages(AllCellsImages <= CellThreshold & AllNucleiImages < NucThreshold));
        NucleiBGCellFGMean = 256*mean(AllNucleiImages(AllNucleiImages <= NucThreshold & AllCellsImages > CellThreshold));
        CellsFGNucBGMean = 256*mean(AllCellsImages(AllCellsImages > CellThreshold & AllNucleiImages < NucThreshold));
        NucleiFGCellAllMean = 256*mean(AllNucleiImages(AllNucleiImages > NucThreshold));
        CellsAllNucFGMean = 256*mean(AllCellsImages(AllNucleiImages > NucThreshold));

        close(handle1);

        %%% Save the interpolated peaks to the handles structure
        handles.Pipeline.NucleiPeak = [NucleiFGCellAllMean;CellsAllNucFGMean];
        handles.Pipeline.CellsPeak = [NucleiBGCellFGMean;CellsFGNucBGMean];
        handles.Pipeline.BackgroundPeak = [NucleiBGCellBGMean;CellsBGNucBGMean];
        
        %%% Finds secondary medians for each half of the actin stained image
        %%% (only used in "hacky" phi, hence the string to match)
        if strcmp(PhiVersion,'hacky')
            handles.Pipeline.SecCellThreshHigh = 256*median(AllCellsImages(AllCellsImages > CellThreshold));
            handles.Pipeline.SecCellThreshLow = 256*median(AllCellsImages(AllCellsImages <= CellThreshold));
        end
        
    end
    
    if ~strfind(PeakSelectionMethod,'Per Image')
        %%% Determines image-wide bias along one axis--that is, if the range of
        %%% intensity for DNA-stained pixels is half that for actin-stained
        %%% pixels, then the messages for nuclei will be half as strong, which
        %%% skews results.  Phi corrects this discrepancy by equalizing along
        %%% the axis with a smaller range of values.
        handles.Pipeline.NucleiMDiff = handles.Pipeline.NucleiPeak(1) - (handles.Pipeline.BackgroundPeak(1) + handles.Pipeline.CellsPeak(1))/2;
        handles.Pipeline.CellsMDiff = handles.Pipeline.CellsPeak(2) - handles.Pipeline.BackgroundPeak(2);
    end

    %%% Saves the preset psi function to handles structure
    handles.Pipeline.Psi = [.9004,.0203,0;.0996,.9396,.0186;0,.0400,.9814];

    drawnow
end

%%% Reads (opens) the images you want to analyze and assigns them variables
OrigNucleiImage = CPretrieveimage(handles,NucleiImageName,ModuleName,'MustBeGray','CheckScale');
OrigCellsImage = CPretrieveimage(handles,CellsImageName,ModuleName,'MustBeGray','CheckScale');
%%% Makes sure neither image is binary
if islogical(OrigCellsImage) || islogical(OrigNucleiImage)
    error(['Image processing was canceled in the ', ModuleName, ' module because the input image is binary (black/white). The input image must be grayscale.']);
end


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Loads and applies illumination correction matrices if specified
if ~(strcmpi(LoadedNIllumCorrName,'none') || strcmpi(LoadedNIllumCorrName,'Do not load')...
        || strcmpi(LoadedCIllumCorrName,'none') || strcmpi(LoadedCIllumCorrName,'Do not load'))
    NucleiCorrMat = CPretrieveimage(handles,LoadedNIllumCorrName,ModuleName,'DontCheckColor','DontCheckScale');
    CellsCorrMat = CPretrieveimage(handles,LoadedCIllumCorrName,ModuleName,'DontCheckColor','DontCheckScale');
    OrigNucleiImage = OrigNucleiImage ./ NucleiCorrMat;
    OrigCellsImage = OrigCellsImage ./ CellsCorrMat;
end

if strcmp(PeakSelectionMethod,'Automatic - Per Image')
    
    %%% For each input image, sets to a valid scale, gets the log of each
    %%% pixel, rescales to 0-1, and stretches data to ideal range
    JLNucImage = imageLog(OrigNucleiImage);
    JLCellImage = imageLog(OrigCellsImage);
    
    %%% Defaults to Otsu's method, ignores potential mask
    NucThreshold = graythresh(JLNucImage);
    CellThreshold = graythresh(JLCellImage);
    
    %%% Gets the means in each section of the image that we believe to have
    %%% a certain pixel label
    NucleiBGCellBGMean = 256*mean(JLNucImage(JLNucImage <= NucThreshold & JLCellImage <= CellThreshold));
    CellsBGNucBGMean = 256*mean(JLCellImage(JLCellImage <= CellThreshold & JLNucImage < NucThreshold));
    NucleiBGCellFGMean = 256*mean(JLNucImage(JLNucImage <= NucThreshold & JLCellImage > CellThreshold));
    CellsFGNucBGMean = 256*mean(JLCellImage(JLCellImage > CellThreshold & JLNucImage < NucThreshold));
    NucleiFGCellAllMean = 256*mean(JLNucImage(JLNucImage > NucThreshold));
    CellsAllNucFGMean = 256*mean(JLCellImage(JLNucImage > NucThreshold));
    
    %%% Save the interpolated peaks to the handles structure
    handles.Pipeline.NucleiPeak = [NucleiFGCellAllMean;CellsAllNucFGMean];
    handles.Pipeline.CellsPeak = [NucleiBGCellFGMean;CellsFGNucBGMean];
    handles.Pipeline.BackgroundPeak = [NucleiBGCellBGMean;CellsBGNucBGMean];
    
    %%% Finds secondary medians for each half of the actin stained image
    %%% (only used in "hacky" phi, hence the string to match)
    if strcmp(PhiVersion,'hacky')
        handles.Pipeline.SecCellThreshHigh = 256*median(JLCellImage(JLCellImage > CellThreshold));
        handles.Pipeline.SecCellThreshLow = 256*median(JLCellImage(JLCellImage <= CellThreshold));
    end
    
    %%% Determines image-wide bias along one axis--that is, if the range of
    %%% intensity for DNA-stained pixels is half that for actin-stained
    %%% pixels, then the messages for nuclei will be half as strong, which
    %%% skews results.  Phi corrects this discrepancy by equalizing along
    %%% the axis with a smaller range of values.
    handles.Pipeline.NucleiMDiff = handles.Pipeline.NucleiPeak(1) - (handles.Pipeline.BackgroundPeak(1) + handles.Pipeline.CellsPeak(1))/2;
    handles.Pipeline.CellsMDiff = handles.Pipeline.CellsPeak(2) - handles.Pipeline.BackgroundPeak(2);
    
end

%%% Log transforms the input images, if not already done, and pads them with
%%% zeros so that messages can be passed in each direction from all pixels,
%%% including those on the border
LoggedPaddedImage = zeros(size(OrigNucleiImage,1)+2,size(OrigNucleiImage,2)+2,2);
if ~exist('JLNucImage','var') || ~exist('JLCellImage','var')
    LoggedPaddedImage(2:end-1,2:end-1,1) = 256*imageLog(OrigNucleiImage);
    LoggedPaddedImage(2:end-1,2:end-1,2) = 256*imageLog(OrigCellsImage);
else
    LoggedPaddedImage(2:end-1,2:end-1,1) = 256*JLNucImage;
    LoggedPaddedImage(2:end-1,2:end-1,2) = 256*JLCellImage;
end    

%%% Creates 4 message-holders for the updating message vectors
Messages.Right = ones(numel(LoggedPaddedImage),3);
Messages.Left= ones(numel(LoggedPaddedImage),3);
Messages.Up = ones(numel(LoggedPaddedImage),3);
Messages.Down = ones(numel(LoggedPaddedImage),3);

%%% Initializes the sub2ind storage, calculates all phi values (these two
%%% steps VASTLY improve runtime by eliminating thousands of subfunction
%%% invocations)
IndicesArray = initsub2ind(size(LoggedPaddedImage));
if strcmpi(PhiVersion,'hacky')
    AllPhiValues = phiH(LoggedPaddedImage,handles);
else
    AllPhiValues = phi(LoggedPaddedImage,handles);
end

if strcmp(TestingMode,'No')

    %%% Runs through the belief propagation algorithm, iterating in each
    %%% direction several times
    PsiFunction = handles.Pipeline.Psi;
    for i=1:NumberOfProps
        Messages = Propagate(LoggedPaddedImage,PsiFunction,AllPhiValues,IndicesArray,Messages);
    end
    drawnow

    %% Calculates beliefs based on these messages, stores them as normalized
    %% double values (where each message vector sums to 1) and as logicals,
    %% where each vector contains one 1 and two 0's
    AllNormalizedBeliefs = zeros(size(OrigNucleiImage,1),size(OrigNucleiImage,2),3);
    AllBeliefs = zeros(size(OrigNucleiImage,1),size(OrigNucleiImage,2));
    LPISize = size(LoggedPaddedImage);
    x = 2:LPISize(2)-1;
    for yind = 2:LPISize(1)-1;
        RawPixelBeliefs = Messages.Up(IndicesArray(yind+1,x),:)' .* ...
            Messages.Down(IndicesArray(yind-1,x),:)' .* ...
            Messages.Left(IndicesArray(yind,x+1),:)' .* ...
            Messages.Right(IndicesArray(yind,x-1),:)' .* ...
            permute(AllPhiValues(yind-1,x-1,:),[3 2 1]);
        NormalizedPixelBeliefs = RawPixelBeliefs ./ repmat(sum(RawPixelBeliefs),3,1);
        [ignore, MaxIndices] = max(NormalizedPixelBeliefs); %#ok Ignore MLint
        for i=1:3
            AllNormalizedBeliefs(yind-1,x-1,i) = reshape(NormalizedPixelBeliefs(i,:),1,size(x,2),1);
        end
        AllBeliefs(yind-1,x-1) = MaxIndices;
    end

else
    %%% TestingMode results in 4 more images being saved to the handles
    %%% structure, each one the beliefs calculated after passing messages
    %%% in each direction during the final propagation.  These are in
    %%% handles.Pipeline.BeliefsAfterPass<1-4> as an array of 0's, 1/2's,
    %%% and 1's representing BG, Cell, and Nucleus, respectively.
    %%% During testing mode, this module can be used normally, but you
    %%% should really only run it when saving or otherwise using the
    %%% intermediary belief calculations, since it slows the module down
    %%% significantly.  
    PsiFunction = handles.Pipeline.Psi;
    for i = 1:NumberOfProps
        Messages = PassUp(LoggedPaddedImage,PsiFunction,AllPhiValues,IndicesArray,Messages);
        if i == NumberOfProps
            [AllNormalizedBeliefs,AllBeliefs] = CalculateBeliefs(size(OrigNucleiImage),size(LoggedPaddedImage),IndicesArray,AllPhiValues,Messages); %#ok
            fieldname = ['BeliefsAfterPass',int2str(1)];
            handles.Pipeline.(fieldname) = (AllBeliefs-1)/2;
        end
        drawnow
        Messages = PassDown(LoggedPaddedImage,PsiFunction,AllPhiValues,IndicesArray,Messages);
        if i == NumberOfProps
            [AllNormalizedBeliefs,AllBeliefs] = CalculateBeliefs(size(OrigNucleiImage),size(LoggedPaddedImage),IndicesArray,AllPhiValues,Messages); %#ok
            fieldname = ['BeliefsAfterPass',int2str(2)];
            handles.Pipeline.(fieldname) = (AllBeliefs-1)/2;
        end
        drawnow
        Messages = PassRight(LoggedPaddedImage,PsiFunction,AllPhiValues,IndicesArray,Messages);
        if i == NumberOfProps
            [AllNormalizedBeliefs,AllBeliefs] = CalculateBeliefs(size(OrigNucleiImage),size(LoggedPaddedImage),IndicesArray,AllPhiValues,Messages); %#ok
            fieldname = ['BeliefsAfterPass',int2str(3)];
            handles.Pipeline.(fieldname) = (AllBeliefs-1)/2;
        end
        drawnow
        Messages = PassLeft(LoggedPaddedImage,PsiFunction,AllPhiValues,IndicesArray,Messages);
        if i == NumberOfProps
            [AllNormalizedBeliefs,AllBeliefs] = CalculateBeliefs(size(OrigNucleiImage),size(LoggedPaddedImage),IndicesArray,AllPhiValues,Messages); %#ok
            fieldname = ['BeliefsAfterPass',int2str(4)];
            handles.Pipeline.(fieldname) = (AllBeliefs-1)/2;
        end
    end
end

%%% Creates binary belief matrices for each pixel label
FinalBinaryNuclei = zeros(size(AllBeliefs));
FinalBinaryNuclei(AllBeliefs==1) = 1;
FinalBinaryCells = zeros(size(AllBeliefs));
FinalBinaryCells(AllBeliefs==2) = 1;
FinalBinaryBackground = zeros(size(AllBeliefs));
FinalBinaryBackground(AllBeliefs==3) = 1;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Doesn't display anything if the figure is closed
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)

    %%% Activates the appropriate figure window
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigNucleiImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the original image
    %%% of nuclei.
    subplot(2,2,1);
    CPimagesc(OrigNucleiImage,handles);
    title(['Input Nuclei Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the original image
    %%% of cells.
    subplot(2,2,3);
    CPimagesc(OrigCellsImage,handles);
    title('Input Cells Image');
    %%% A subplot of the figure window is set to display all labeling
    %%% results
    TempAllBeliefs = AllBeliefs;
    TempAllBeliefs(1,1:3) = 1:3;
    subplot(2,2,2);
    CPimagesc(TempAllBeliefs,handles);
    title('Output');
    %%% A 'subplot' of the figure window is set to display the
    %%% user-selected or input intensity peaks
    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','fontname','helvetica','units','normalized','position',[0.5 0 0.5 0.4],'backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize+4);
    set(displaytexthandle,'string',sprintf(['Peak Intensity Values as (DNA-X,Actin-Y)\n\nNuclei: (%.1f, %.1f)',...
        '\nCells: (%.1f, %.1f)\nBackground: (%.1f, %.1f)'],handles.Pipeline.NucleiPeak,handles.Pipeline.CellsPeak,handles.Pipeline.BackgroundPeak));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if ~strcmpi(NucleiOutputName,'Do not save')
    handles.Pipeline.(NucleiOutputName) = FinalBinaryNuclei;
end
if ~strcmpi(CellsOutputName,'Do not save')
    handles.Pipeline.(CellsOutputName) = FinalBinaryCells|FinalBinaryNuclei;
end
if ~strcmpi(BackgroundOutputName,'Do not save')
    handles.Pipeline.(BackgroundOutputName) = FinalBinaryBackground;
end

%%% Uncomment these lines to save the intial phi values before message
%%% passing and the final belief probabilities (before rounding off to
%%% create the binaries)
% handles.Pipeline.BPInitialPhiLabels = AllPhiValues;
% handles.Pipeline.BPProbableBeliefMatrix = AllNormalizedBeliefs;
%%%

if ~strcmpi(GrayscaleOutputName,'Do not save')
    handles.Pipeline.(GrayscaleOutputName) = (AllBeliefs-1)/2;
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function arr = initsub2ind(inputsize)
%%% Returns the array of all single-term indices for an array of size
%%% inputsize, to be indexed instead of calling sub2ind over and over
arr = zeros(inputsize);
if size(inputsize,2) == 3
    z = inputsize(3);
else z = 1;
end
xs = 1:inputsize(2);
for zind = 1:z
    zs = zind*ones(size(xs));
    for yind = 1:inputsize(1)
        ys = yind*ones(size(xs));
        arr(yind,xs,zind) = sub2ind(inputsize,ys,xs,zs);
    end
end
%%% to get the equivalent of sub2ind(y,x), get arr(yind,x) or arr(y,xind)'

function Messages = Propagate(padim,psi,allphivals,indsarr,Messages)
%%% Bundles together the passing functions, propagating messages throughout
%%% the image, simulating loopy propagation by treating the image at each
%%% step as a tree (a directed graph with no cliques)
Messages = PassLeft(padim,psi,allphivals,indsarr,...
    PassRight(padim,psi,allphivals,indsarr,...
    PassDown(padim,psi,allphivals,indsarr,...
    PassUp(padim,psi,allphivals,indsarr,Messages))));

function Messages = PassUp(padim,psi,allphivals,indsarr,Messages)
%%% For all pixels in a padded image, calculates the message to pass up
%%% from that location and store it in Messages.Up
%%% padim should be a NxMx2 image, where N and M are 2 larger than the
%%% respective dimensions of the original input images, and the two sheets
%%% contain corresponding pixel intensity values for DNA and actin
%%% staining; psi should be a 3x3 psi function which relates the
%%% probabilities that a pixel will have each labels based only on what its
%%% neighbors are labeled; allphivals is the array calculated by phi
%%% containing the probability that each pixel will be labeled in each
%%% category based on its brightness in two channels; and Messages is a 1x1
%%% struct containing 4 N*Mx3 arrays that hold all updating messages passed
%%% from pixel to pixel
x = 2:size(padim,2)-1;
%%% for each row, process the entire column (this is vectorized form that
%%% takes sqrt as long as the original, nonvectorized nested loops)
for yind = size(padim,1)-1:-1:2
    %%% saves the messages coming into each pixel in (yind,x) - from the
    %%% pixels to the left going Right, from the right going Left, and from
    %%% below going Up
    rmsgs = Messages.Right(indsarr(yind,x-1),:)';
    lmsgs = Messages.Left(indsarr(yind,x+1),:)';
    umsgs = Messages.Up(indsarr(yind+1,x),:)';
    %%% gets the product of all incoming messages, multiplies this by the
    %%% phi values for these pixels, and passes the result through the psi
    %%% function, then normalizes so each column sums to 1
    prelimmessages = psi*(permute(allphivals(yind-1,x-1,:),[3 2 1]).*rmsgs.*lmsgs.*umsgs);
    messages = prelimmessages./repmat(sum(prelimmessages),3,1);
    %%% stores these messages in Messages.Up
    Messages.Up(indsarr(yind,x),:) = messages';
end

function Messages = PassDown(padim,psi,allphivals,indsarr,Messages)
%%% Updates the messages to pass down from each pixel -- see PassUp for
%%% more complete documentation
x = 2:size(padim,2)-1;
for yind = 2:size(padim,1)-1
    rmsgs = Messages.Right(indsarr(yind,x-1),:)';
    lmsgs = Messages.Left(indsarr(yind,x+1),:)';
    dmsgs = Messages.Down(indsarr(yind-1,x),:)';
    prelimmessages = psi*(permute(allphivals(yind-1,x-1,:),[3 2 1]).*rmsgs.*lmsgs.*dmsgs);
    messages = prelimmessages./repmat(sum(prelimmessages),3,1);
    Messages.Down(indsarr(yind,x),:) = messages';
end

function Messages = PassLeft(padim,psi,allphivals,indsarr,Messages)
%%% Updates the messages to pass left from each pixel -- see PassUp for
%%% more complete documentation
y = 2:size(padim,1)-1;
for xind = size(padim,2)-1:-1:2
    lmsgs = Messages.Left(indsarr(y,xind+1),:)';
    umsgs = Messages.Up(indsarr(y+1,xind),:)';
    dmsgs = Messages.Down(indsarr(y-1,xind),:)';
    prelimmessages = psi*(permute(allphivals(y-1,xind-1,:),[3 1 2]).*lmsgs.*dmsgs.*umsgs);
    messages = prelimmessages./repmat(sum(prelimmessages),3,1);
    Messages.Left(indsarr(y,xind),:) = messages';
end

function Messages = PassRight(padim,psi,allphivals,indsarr,Messages)
%%% Updates the messages to pass right from each pixel -- see PassUp for
%%% more complete documentation
y = 2:size(padim,1)-1;
for xind = 2:size(padim,2)-1
    rmsgs = Messages.Right(indsarr(y,xind-1),:)';
    umsgs = Messages.Up(indsarr(y+1,xind),:)';
    dmsgs = Messages.Down(indsarr(y-1,xind),:)';
    prelimmessages = psi*(permute(allphivals(y-1,xind-1,:),[3 1 2]).*rmsgs.*dmsgs.*umsgs);
    messages = prelimmessages./repmat(sum(prelimmessages),3,1);
    Messages.Right(indsarr(y,xind),:) = messages';
end

function arr = phi(padim,handles)
%%% returns an array containing phi values (1x1x3) at each pixel in padim
%%% except the border, as a R-1xC-1x3 array where padim is RxCx2

%%% Initializes counters, preallocates the array to return
rows = size(padim,1)-2;
cols = size(padim,2)-2;
arr = zeros(rows,cols,3);

%%% Extracts relevant information from the handles structure
c = repmat(handles.Pipeline.CellsPeak,1,cols);
n = repmat(handles.Pipeline.NucleiPeak,1,cols);
b = repmat(handles.Pipeline.BackgroundPeak,1,cols);
scaling = [1/handles.Pipeline.NucleiMDiff 0; 0 1/handles.Pipeline.CellsMDiff];
sigma = handles.Pipeline.SigmaValue;
actinsc = [1 0; 0 1/handles.Pipeline.ActinScalingFactor];
% chi = handles.Pipeline.SecCellThreshHigh;
% clo = handles.Pipeline.SecCellThreshLow;

%%% for each row, for each column within that row, calculates the
%%% probability that a given pixel will be labeled in each of the three
%%% categories based on only its pixel intensity values.
for yind = 1:rows
    
    %%% x is the array of pixel values, [DNA;actin], for each corresponding
    %%% pixel in padim, accounting for the pad of zeros
    x = [padim(yind+1,2:end-1,1);padim(yind+1,2:end-1,2)];
    % %%% Finds the locations (single-subscript indexing) where actin stain
    % %%% intensities are above, below, and between means of each half
    % lowestlocs = find(x(2,:) < clo);
    % highestlocs = find(x(2,:) > chi);
    % restoflocs = find(x(2,:) > clo & x(2,:) < chi);
    
    %%% Calculates probabilities of each label by finding distances, fixing
    %%% the actin staining data according to secondary means, and putting
    %%% this into a gaussian probability function for each label
    sdB = scaling * (b-x);
    sdN = scaling * (n-x);
    sdC = actinsc * scaling * (c-x);
    % sdC = scaling * (c-x);
    % sdC(2,lowestlocs) = 0.9;
    % sdC(2,highestlocs) = 0.1;
    % sdC(2,restoflocs) = 0.4;
    % sdB(2,lowestlocs) = 0.1;
    % sdB(2,highestlocs) = 0.9;
    % sdB(2,restoflocs) = 0.6;
    invdisB = exp(sum(-sdB.*sdB/sigma));
    invdisN = exp(sum(-sdN.*sdN/sigma));
    invdisC = exp(sum(-sdC.*sdC/sigma));
    
    %%% Sums these values and normalizes so that they sum to 1
    z=invdisB+invdisC+invdisN;
    probB=invdisB./z;
    probN=invdisN./z;
    probC=invdisC./z;
    %%% stores the results (which are an Nx3 array) into a 1xNx3 slice of
    %%% the results array
    arr(yind,:,:) = permute([probN; probC; probB],[3 2 1]);
    
end

function arr = phiH(padim,handles)
%%% returns an array containing phi values (1x1x3) at each pixel in padim
%%% except the border, as a R-1xC-1x3 array where padim is RxCx2
%%% MODDED CURRENTLY TO BE THE 'HACKY' VERSION, MEANING THAT VALUES ARE
%%% HARD-CODED FOR ACTIN-INFO IN CELL,BG DISTANCES
CPhelpdlg('Alert: Using "hacky" version of the phi subfunction!');

%%% Initializes counters, preallocates the array to return
rows = size(padim,1)-2;
cols = size(padim,2)-2;
arr = zeros(rows,cols,3);

%%% Extracts relevant information from the handles structure
c = repmat(handles.Pipeline.CellsPeak,1,cols);
n = repmat(handles.Pipeline.NucleiPeak,1,cols);
b = repmat(handles.Pipeline.BackgroundPeak,1,cols);
scaling = [1/handles.Pipeline.NucleiMDiff 0; 0 1/handles.Pipeline.CellsMDiff];
sigma = handles.Pipeline.SigmaValue;
% actinsc = [1 0; 0 1/handles.Pipeline.ActinScalingFactor];
chi = handles.Pipeline.SecCellThreshHigh;
clo = handles.Pipeline.SecCellThreshLow;

%%% for each row, for each column within that row, calculates the
%%% probability that a given pixel will be labeled in each of the three
%%% categories based on only its pixel intensity values.
for yind = 1:rows
    
    %%% x is the array of pixel values, [DNA;actin], for each corresponding
    %%% pixel in padim, accounting for the pad of zeros
    x = [padim(yind+1,2:end-1,1);padim(yind+1,2:end-1,2)];
    %%% Finds the locations (single-subscript indexing) where actin stain
    %%% intensities are above, below, and between means of each half
    lowestlocs = find(x(2,:) < clo);
    highestlocs = find(x(2,:) > chi);
    restoflocs = find(x(2,:) > clo & x(2,:) < chi);
    
    %%% Calculates probabilities of each label by finding distances, fixing
    %%% the actin staining data according to secondary means, and putting
    %%% this into a gaussian probability function for each label
    sdB = scaling * (b-x);
    sdN = scaling * (n-x);
    % sdC = actinsc * scaling * (c-x);
    sdC = scaling * (c-x);
    sdC(2,lowestlocs) = 0.9;
    sdC(2,highestlocs) = 0.1;
    sdC(2,restoflocs) = 0.4;
    sdB(2,lowestlocs) = 0.1;
    sdB(2,highestlocs) = 0.9;
    sdB(2,restoflocs) = 0.6;
    invdisB = exp(sum(-sdB.*sdB/sigma));
    invdisN = exp(sum(-sdN.*sdN/sigma));
    invdisC = exp(sum(-sdC.*sdC/sigma));
    
    %%% Sums these values and normalizes so that they sum to 1
    z=invdisB+invdisC+invdisN;
    probB=invdisB./z;
    probN=invdisN./z;
    probC=invdisC./z;
    %%% stores the results (which are an Nx3 array) into a 1xNx3 slice of
    %%% the results array
    arr(yind,:,:) = permute([probN; probC; probB],[3 2 1]);
    
end

function [allnormbeliefs, allbeliefs] = CalculateBeliefs(sizeOrig,sizePadded,indsarr,phivals,messages)
%%% Calculates beliefs based on the current state of messages (at any
%%% passing moment) and returns them as an array of 1-2-3 and as their
%%% final probabilities
 
allnormbeliefs = zeros(sizeOrig(1),sizeOrig(2),3);
allbeliefs = zeros(sizeOrig(1),sizeOrig(2));
x = 2:sizePadded(2)-1;
%LPISize = size(LoggedPaddedImage);
for yind = 2:sizePadded(1)-1;
    rawPixelBeliefs = messages.Up(indsarr(yind+1,x),:)' .* ...
        messages.Down(indsarr(yind-1,x),:)' .* ...
        messages.Left(indsarr(yind,x+1),:)' .* ...
        messages.Right(indsarr(yind,x-1),:)' .* ...
        permute(phivals(yind-1,x-1,:),[3 2 1]);
    normalizedPixelBeliefs =rawPixelBeliefs ./ repmat(sum(rawPixelBeliefs),3,1);
    [ignore, maxIndices] = max(normalizedPixelBeliefs); %#ok Ignore MLint
    for i=1:3
        allnormbeliefs(yind-1,x-1,i) = reshape(normalizedPixelBeliefs(i,:),1,size(x,2),1);
    end
    allbeliefs(yind-1,x-1) = maxIndices;
end

function im = imageLog(im)
%%% Performs the standard range-adjustment, log-transform, and
%%% normalization for images (of input scale 0-1)

%%% Sets pixels below max(min(im(:)),1/(2^12)) to that minimum so we have a
%%% valid dynamic scale of pixel values and can work with 12/16-bit DIB's
%%% and regular 8-bit images
imnozeros = im;
imnozeros(imnozeros == 0) = 2;
minval = max(min(imnozeros(:)),1/(2^12));
clampedim = im;
clampedim(clampedim < minval) = minval;
%%% Gets the log of the image
loggedim = log(clampedim);
%%% Normalizes, resets to 0-1 scale (stretching in the process if
%%% necessary)
minval = min(loggedim(:));
maxval = max(loggedim(:));
im = (loggedim - minval) / (maxval - minval);
