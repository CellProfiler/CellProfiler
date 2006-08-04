function handles = DistinguishPixelLabels(handles)

% Help for the Distinguish Pixel Labels module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Given 2 input images, labels all pixels as cell, nucleus, or background
% using belief propagation.
% *************************************************************************
%
% BETA VERSION
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

% Currently being developed by Chris Gang
% TODO
% - eventually a more advanced phi function would work better - one that
%   fits gaussian blobs to the 2D histogram and uses something more
%   advanced than distance from peaks to calculate probability
% - it doesn't really make sense to require that the histogram or
%   correction matrices be calculated ahead of time
% - working on the automatic, per-image calculation of each type of 
%   intensity peaks - display needs to be fixed, and why do no/very 
%   few pixels get labeled as nuclei?


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
%defaultVAR05 = BPBackground
BackgroundOutputName = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%infotypeVAR05 = imagegroup indep

%textVAR06 = Choose peak pixel intensity selection method:
%choiceVAR06 = Automatic - Per Image
%choiceVAR06 = Numeric - All
%choiceVAR06 = Mouse - All
PeakSelectionMethod = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = For NUMERIC, enter the coordinates for the peak in nucleus pixel intensity values as (NucleiX,NucleiY).
%defaultVAR07 = 100,100
NucleiPeakInputValues = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For NUMERIC, enter the coordinates for the peak in cell pixel intensity values as (CellsX,CellsY).
%defaultVAR08 = 150,150
CellsPeakInputValues = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For NUMERIC, enter the coordinates for the peak in background pixel intensity values as (BGX,BGY).
%defaultVAR09 = 200,200
BackgroundPeakInputValues = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For MOUSE, you must either load two illumination correction matrices and calculate the 2-dimensional intensity histogram OR load the histogram file you saved earlier.  If you don't load a histogram, one will be calculated for all input cell and nucleus images during this module's first cycle.  Choose which file to load:
%choiceVAR10 = Histogram
%choiceVAR10 = Correction Matrices
MouseInputMethod = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = For MOUSE and HISTOGRAM, what did you call the 2-dimensional intensity histogram file? (Use LoadSingleImage to load a .mat file containing a variable called "Image".)
%infotypeVAR11 = imagegroup
LoadedHistogramName = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = For MOUSE and CORRECTION MATRICES, what did you call the illumination correction matrix for nuclei? Choose "Other..." and enter "none" if you do not wish to correct illumination. (Use LoadSingleImage to load a .mat file containing a variable called "Image".)
%infotypeVAR12 = imagegroup
LoadedNIllumCorrName = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 = For MOUSE and CORRECTION MATRICES, what did you call the illumination correction matrix for cells? Choose "Other..." and enter "none" if you do not wish to correct illumination. (Use LoadSingleImage to load a .mat file containing a variable called "Image".)
%infotypeVAR13 = imagegroup
LoadedCIllumCorrName = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu custom

%textVAR14 = For MOUSE and CORRECTION MATRICES, what do you want to call the histogram calculated using all images and these correction matrices (optional)?
%defaultVAR14 = Do not save
SaveHistogram = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If this is the first image set (and only then), displays or
%%% creates the 2-Dimensional histogram to interactively find the peak
%%% illumination values for cell, nucleus, and background OR loads the
%%% numeric, preset peak illumination values
if handles.Current.SetBeingAnalyzed == 1

    if strncmp(PeakSelectionMethod,'Numeric',7)
        
        %%% Parses user input for the three points
        NucleiPixel = str2num(NucleiPeakInputValues); %#ok ignore MLint
        NxDNA = NucleiPixel(1);
        NyActin = NucleiPixel(2);
        CellsPixel = str2num(CellsPeakInputValues); %#ok ignore MLint
        CxDNA = CellsPixel(1);
        CyActin = CellsPixel(2);
        BackgroundPixel = str2num(BackgroundPeakInputValues); %#ok ignore MLint
        BGxDNA = BackgroundPixel(1);
        BGyActin = BackgroundPixel(2);
        
    elseif strncmp(PeakSelectionMethod,'Mouse',5)
        %%% The user will be interactively selecting the peak illumination
        %%% values by clicking on a histogram
        if strcmp(MouseInputMethod,'Histogram')
            %%% The histogram is already loaded, so reads (opens) the file
            %%% to be shown later on
            SumHistogram = CPretrieveimage(handles,LoadedHistogramName,ModuleName,'DontCheckColor','DontCheckScale');
            
        elseif strcmp(MouseInputMethod,'Correction Matrices')
            %%% The histogram hasn't yet been calculated, so we must load
            %%% the correction matrices and preprocess the entire image set
            %%% to create a histogram, from which we'll get the peaks of
            %%% intensity for each label.
            
            %%% Reads (opens) the correction matrix files as specified in
            %%% LoadSingleImage before this module
            if (strcmpi(LoadedNIllumCorrName,'none') || strcmpi(LoadedNIllumCorrName,'Do not load'))...
                    && (strcmpi(LoadedCIllumCorrName,'none') || strcmpi(LoadedCIllumCorrName,'Do not load'))
                CorrectIllumination = 0;
            else
                NucleiCorrMat = CPretrieveimage(handles,LoadedNIllumCorrName,ModuleName,'DontCheckColor','DontCheckScale');
                CellsCorrMat = CPretrieveimage(handles,LoadedCIllumCorrName,ModuleName,'DontCheckColor','DontCheckScale');
                CorrectIllumination = 1;
            end
            
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
            
            %%% Initializes the sparse matrix representing the sum of all
            %%% intensity histograms
            SumHistogram = sparse(256,256);
            handle1 = CPhelpdlg(['Preliminary calculations are under way for the ', ModuleName, ' module.  Subsequent cycles skip this step and will run much more quickly.']);
            StillOpen = 1;
            PrelimStartTime = toc;
            
            %%% Loops through all loaded images during this first cycle
            %%% (assumes that there are an equal number of cell and nucleus
            %%% images)
            for i=1:length(NucleiFileList)

                %%% Updates a help dialog with progress information every
                %%% time 10% of images are preprocessed
                if floor(mod(i-1,length(NucleiFileList)/10)) == 0 && i~= 1
                    if StillOpen % first time this appears
                        close(handle1);
                        StillOpen = 0;
                    end
                    PrelimElapsedTime = toc - PrelimStartTime;
                    EstRemaining = PrelimElapsedTime*((length(NucleiFileList)/(i-1))-1);
                    handle1 = CPhelpdlg(['Loading image set #', int2str(i), ' for preliminary histogram calculation (', int2str(100*(i-1)/length(NucleiFileList)), '% complete).  ',...
                        'Estimated time remaining is ',num2str(EstRemaining/60), ' minutes.'],...
                        ['Preliminary calculations for ',ModuleName]);
                end
                drawnow
                [LoadedNucleiImage, handles] = CPimread(fullfile(NucleiPathname,char(NucleiFileList(i))),handles);
                [LoadedCellsImage, handles] = CPimread(fullfile(CellsPathname,char(CellsFileList(i))),handles);
                %%% Jitters each image (adding a random fraction based on
                %%% the images' bit depth) so we can get the log without
                %%% skewing data
                %JitteredNucleiImage = LoadedNucleiImage + rand(size(LoadedNucleiImage)) / 256;
                %JitteredCellsImage = LoadedCellsImage + rand(size(LoadedCellsImage)) / 256;
                %%% Divides by the illumination correction factor matrices
                if CorrectIllumination
                    CorrectedNucleiImage = LoadedNucleiImage ./ NucleiCorrMat;
                    CorrectedCellsImage = LoadedCellsImage ./ CellsCorrMat;
                else
                    CorrectedNucleiImage = LoadedNucleiImage;
                    CorrectedCellsImage = LoadedCellsImage;
                end
                %%% "Clamps" any pixels that remain below the minimum
                %%% allowed pixel value, set based on the images' bit depth
                MinimumPixVal = 1/256;
                ClampedNucleiImage = CorrectedNucleiImage;
                ClampedCellsImage = CorrectedCellsImage;
                ClampedNucleiImage(CorrectedNucleiImage < MinimumPixVal) = MinimumPixVal;
                ClampedCellsImage(CorrectedCellsImage < MinimumPixVal) = MinimumPixVal;
                
                JitteredNucleiImage = log(ClampedNucleiImage) + rand(size(ClampedNucleiImage)) .*...
                    (log(ClampedNucleiImage + MinimumPixVal) - log(ClampedNucleiImage));
                JitteredCellsImage = log(ClampedCellsImage) + rand(size(ClampedCellsImage)) .*...
                    (log(ClampedCellsImage + MinimumPixVal) - log(ClampedCellsImage));
                %%% Log-transforms the images, normalizes to a scale of
                %%% 0-1, rescales by multiplication to 0-255, adds 1, and
                %%% rounds down (the effect of this is to make the pixel
                %%% values of each image histogrammable, with 256 bins)
                BoxedNucleiImage = floor(255 * (JitteredNucleiImage - log(MinimumPixVal)) ...
                    / (-log(MinimumPixVal)) + 1);
                BoxedCellsImage = floor(255 * (JitteredCellsImage - log(MinimumPixVal)) ...
                    / (-log(MinimumPixVal)) + 1);
                
                %%% Creates a 2-D histogram, represented in sparse matrix
                %%% form by taking advantage of sparse's ability to add the
                %%% values in those locations where pixel values repeat.
                %%% Then, adds this histogram to the overall histogram of
                %%% all images.
                ThisIterHistogram = sparse(BoxedNucleiImage,BoxedCellsImage,1,256,256);
                SumHistogram = SumHistogram + ThisIterHistogram;
                
            end
            close(handle1);
            handles.Pipeline.HistogramToDisplay = SumHistogram;
            
        else
            error(['Image processing was canceled in the ',ModuleName,' module because, somehow, the method you selected for interactive selection of the pixel intensity peaks (',MouseInputMethod,') is invalid.']);
        end
        
        %%% Displays the histogram in this module's figure and prompts for
        %%% the user to select the three "peaks" of intensity values, then
        %%% stores these values.
        %%% Determines the figure number to display in.
        ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
        FigureHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
        %%% Displays the histogram in the upper 2/3 of the figure, sets
        %%% origin to lower left, uses the desired coloring system, and
        %%% labels the axes
        set(FigureHandle,'units','normalized');
        OldPos = get(FigureHandle,'position');
        set(FigureHandle,'position',[OldPos(1) OldPos(2)-(OldPos(4)*.5) OldPos(3) OldPos(4)*1.5]);
        subplot(3,2,[1 2 3 4]);
        CPimagesc(SumHistogram',handles);
        axis xy;
        axis image;
        set(gcf,'colormap',colormap('jet'));
        ylabel('Actin staining intensity');
        xlabel('DNA staining intensity');
        
        %%% Prompts the user to identify 3 peaks in intensity (& checks
        %%% that input is only one point by looping through each try)
        success = 0;
        retrytext = '';
        while success==0 
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','fontname','helvetica','position',[80 10 400 130],'backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize+2);
            displaytext = 'Select the peak in intensity that is closest to the lower left corner -- low in both DNA and actin stain intensity, it should represent the background pixels.  Click on a point and press Enter to confirm, or just double-click on it.  Press Delete or Backspace to undo your selection, and you can also use MATLAB''s zoom tools if necessary.';
            set(displaytexthandle,'string',[retrytext displaytext])
            drawnow
            [BGxDNA,BGyActin] = getpts(FigureHandle);
            delete(displaytexthandle);
            drawnow
            if isscalar(BGxDNA); success=1;
            else retrytext = 'Please try again, but this time only select one point.  ';
            end
        end
        success = 0;
        retrytext = '';
        while success==0 
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','fontname','helvetica','position',[80 10 400 130],'backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize+2);
            displaytext = 'Now select the intensity peak still to the left but nearer to the top, formed of the pixels representing cells.  If your images contain confluent cells, this is likely to be the highest (reddest) peak by a significant margin. Click on a point and press Enter to confirm, or just double-click on it.  If you need to undo your selection, press Backspace or Delete, and you can also use MATLAB''s zoom tools if necessary.';
            set(displaytexthandle,'string',[retrytext displaytext])
            drawnow
            [CxDNA,CyActin] = getpts(FigureHandle);
            delete(displaytexthandle);
            drawnow
            if isscalar(CxDNA); success=1;
            else retrytext = 'Please try again, but this time only select one point.  ';
            end
        end
        success = 0;
        retrytext = '';
        while success==0 
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','fontname','helvetica','position',[80 10 400 130],'backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize+2);
            displaytext = 'Finally, select the intensity peak farthest to the right, which represents the pixels that form nuclei.  This peak may be less distinct than the cell peak.  Click on a point and press Enter to confirm, or just double-click on it.  If you need to undo your selection, press Backspace or Delete, and you can also use MATLAB''s zoom tools if necessary.';
            set(displaytexthandle,'string',[retrytext displaytext])
            drawnow
            [NxDNA,NyActin] = getpts(FigureHandle);
            delete(displaytexthandle);
            drawnow
            if isscalar(NxDNA); success=1;
            else retrytext = 'Please try again, but this time only select one point.  ';
            end
        end
        subplot(1,1,1);
        set(FigureHandle,'position',OldPos);
        set(FigureHandle,'units','pixels');

    elseif strncmp(PeakSelectionMethod,'Automatic',9)
        %%% do per-image, automatic peak calculation in the IMAGE ANALYSIS
        %%% step - and do nothing here
    else
        error(['Image processing was canceled in the ',ModuleName,' module because, somehow, the method you selected for selecting the pixel intensity peaks (',PeakSelectionMethod,') is invalid.']);
    end
    
    if ~strncmp(PeakSelectionMethod,'Automatic',9)
        %%% Stores the found values (which should be only one pixel each) in
        %%% the handles.Pipeline structure
        handles.Pipeline.CellsPeak = [CyActin;CxDNA];
        handles.Pipeline.NucleiPeak = [NyActin;NxDNA];
        handles.Pipeline.BackgroundPeak = [BGyActin;BGxDNA];
        handles.Pipeline.PeakIntensityString = ...
            sprintf('Peak Intensity Values\n\nNuclei: (%.1f, %.1f)\nCells: (%.1f, %.1f)\nBackground: (%.1f, %.1f)',NxDNA,NyActin,CxDNA,CyActin,BGxDNA,BGyActin);
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

if strncmp(PeakSelectionMethod,'Automatic',9)
    %%% Jitters these images on the log scale
    NucNonZeros = OrigNucleiImage(OrigNucleiImage >= 0);
    CellNonZeros = OrigCellsImage(OrigCellsImage >= 0);
    MinNucPixVal = min(NucNonZeros(:));
    MinCellPixVal = min(CellNonZeros(:));
    JNucImage = log(OrigNucleiImage) + rand(size(OrigNucleiImage)) .*...
        (log(OrigNucleiImage + MinNucPixVal) - log(OrigNucleiImage));
    JCellImage = log(OrigCellsImage) + rand(size(OrigCellsImage)) .*...
        (log(OrigCellsImage + MinCellPixVal) - log(OrigCellsImage));
    
    JLNucImage = (JNucImage - log(MinNucPixVal)) / -log(MinNucPixVal);
    JLCellImage = (JCellImage - log(MinCellPixVal)) / -log(MinCellPixVal);
    
    %%% Gets thresholds for each image, applies them to get a bw screen
    [handles,NucThreshold] = CPthreshold(handles,'Global MoG','50%','Do not use','Do not use',1,JLNucImage,'placeholder',ModuleName);
    [handles,CellThreshold] = CPthreshold(handles,'Global MoG','50%','Do not use','Do not use',1,JLCellImage,'placeholder',ModuleName);
    %NucThreshold = graythresh(JLNucImage);
    %CellThreshold = graythresh(JLCellImage);
    
    %%% Gets the mean of each image for all the foreground and all the
    %%% background pixels separately
    NucleiFGMean = 256*mean(JLNucImage(JLNucImage > NucThreshold));
    NucleiBGMean = 256*mean(JLNucImage(JLNucImage <= NucThreshold));
    CellsFGMean = 256*mean(JLCellImage(JLCellImage > CellThreshold));
    CellsBGMean = 256*mean(JLCellImage(JLCellImage <= CellThreshold));
    %%% Save the interpolated peaks to the handles structure
    handles.Pipeline.NucleiPeak = [CellsFGMean;NucleiFGMean];
    handles.Pipeline.CellsPeak = [CellsFGMean;NucleiBGMean];
    handles.Pipeline.BackgroundPeak = [CellsBGMean;NucleiBGMean];
    handles.Pipeline.PeakIntensityString = ...
        sprintf('Peak Intensity Values\n\nNuclei: (%.1f, %.1f)\nCells: (%.1f, %.1f)\nBackground: (%.1f, %.1f)',...
        NucleiFGMean,CellsFGMean,NucleiBGMean,CellsFGMean,NucleiBGMean,CellsBGMean);
    
    %%% Create the 2D histogram for display purposes
    RoundNucImage = floor(255 * JLNucImage + 1);
    RoundCellImage = floor(255 * JLCellImage + 1);
 %   save('presparse.mat','RoundNucImage','RoundCellImage');
    AutoHistogram = sparse(RoundNucImage,RoundCellImage,1,256,256);
    MargHistDNA = hist(JLNucImage(:),100);
    MargHistActin = hist(JLCellImage(:),100);
    handles.Pipeline.HistogramToDisplay = AutoHistogram;
    
    %%% Save 2d, & both marginal histograms to pipeline (comment this out!)
    fieldname = ['Auto2DHistogramSet',int2str(handles.Current.SetBeingAnalyzed)];
    handles.Pipeline.(fieldname) = AutoHistogram;
    fieldname = ['AutoMarginalHistDNASet',int2str(handles.Current.SetBeingAnalyzed)];
    handles.Pipeline.(fieldname) = MargHistDNA;
    fieldname = ['AutoMarginalHistActinSet',int2str(handles.Current.SetBeingAnalyzed)];
    handles.Pipeline.(fieldname) = MargHistActin;
    
 %   save('autothresh.mat','JLNucImage','JLCellImage','NucThreshold','CellThreshold','AutoHistogram','MargHistDNA','MargHistActin');
 %   error('save me');
end

%%% Scales both input images to 0-256, loads them into 1 3D array with a
%%% padded one-row/col border of zeros
PaddedCompositeImage = zeros(size(OrigNucleiImage,1)+2,size(OrigNucleiImage,2)+2,2);
PaddedCompositeImage(2:end-1,2:end-1,1) = 256*OrigNucleiImage;
PaddedCompositeImage(2:end-1,2:end-1,2) = 256*OrigCellsImage;
%%% Log-transforms the pixel values, keeping 0-256 scale
LoggedPaddedImage = 32 * (log(PaddedCompositeImage+1)/log(2));

%%% Creates 4 message-holders for the updating message vectors
Messages.Right = ones(numel(LoggedPaddedImage),3);
Messages.Left= ones(numel(LoggedPaddedImage),3);
Messages.Up = ones(numel(LoggedPaddedImage),3);
Messages.Down = ones(numel(LoggedPaddedImage),3);

%%% Initializes the sub2ind storage
IndicesArray = initsub2ind(size(LoggedPaddedImage));
AllPhiValues = phi(LoggedPaddedImage,handles);
%AllPhiValues = phi2(LoggedPaddedImage);

%%% Runs through the belief propagation algorithm, iterating in each
%%% direction several times
for i=1:5
%    Messages = Propagate(LoggedPaddedImage,handles,AllPhiValues,IndicesArray,Messages);
end
drawnow

%%% Calculates beliefs based on these messages, stores them as normalized
%%% double values (where each message vector sums to 1) and as logicals,
%%% where each vector contains one 1 and two 0's
AllNormalizedBeliefs = zeros(size(OrigNucleiImage,1),size(OrigNucleiImage,2),3);
AllBeliefs = zeros(size(OrigNucleiImage,1),size(OrigNucleiImage,2));
x = 2:size(LoggedPaddedImage,2)-1;
LPISize = size(LoggedPaddedImage);
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

%%% Creates binary belief matrices for each pixel label
FinalBinaryNuclei = zeros(size(AllBeliefs));
FinalBinaryNuclei(AllBeliefs==1) = 1;
FinalBinaryCells = zeros(size(AllBeliefs));
FinalBinaryCells(AllBeliefs==2) = 1;
FinalBinaryBackground = zeros(size(AllBeliefs));
FinalBinaryBackground(AllBeliefs==3) = 1;

FinalBinaryCellsAndNuc = FinalBinaryCells | FinalBinaryNuclei;
AdjustedAllBeliefs = (AllBeliefs-1)/2;

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
    
    if isfield(handles.Pipeline,'HistogramToDisplay')
        %%% A subplot of the figure window is set to display the histogram
        %%% and the peaks for each label
        subplot(2,2,4);
        CPimagesc(handles.Pipeline.HistogramToDisplay',handles);
        axis xy;
        text(handles.Pipeline.NucleiPeak(2),handles.Pipeline.NucleiPeak(1),'\leftarrow DNA peak','HorizontalAlignment','left','BackgroundColor','w','color','b');
        text(handles.Pipeline.CellsPeak(2),handles.Pipeline.CellsPeak(1),'Actin peak \rightarrow','HorizontalAlignment','right','BackgroundColor','w','color','b');
        text(handles.Pipeline.BackgroundPeak(2),handles.Pipeline.BackgroundPeak(1),'BG peak \rightarrow','HorizontalAlignment','right','BackgroundColor','w','color','b');
        title('Peak Intensity Values');
        CPhelpdlg(handles.Pipeline.PeakIntensityString);
    else
        %%% A 'subplot' of the figure window is set to display the
        %%% user-selected or input intensity peaks
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','fontname','helvetica','units','normalized','position',[0.5 0 0.5 0.4],'backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize+4);
        set(displaytexthandle,'string',handles.Pipeline.PeakIntensityString);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(NucleiOutputName) = FinalBinaryNuclei;
%handles.Pipeline.(CellsOutputName) = FinalBinaryCells;
handles.Pipeline.(CellsOutputName) = FinalBinaryCellsAndNuc;
handles.Pipeline.(BackgroundOutputName) = FinalBinaryBackground;


if handles.Current.SetBeingAnalyzed == 1 && strcmp(MouseInputMethod,'Correction Matrices') && ~strcmpi(SaveHistogram,'Do not save')
    handles.Pipeline.(SaveHistogram) = SumHistogram;
end

handles.Pipeline.BPAbsoluteBeliefMatrix = AllBeliefs;
handles.Pipeline.BPProbableBeliefMatrix = AllNormalizedBeliefs;
handles.Pipeline.BPAdjustedBeliefMatrix = AdjustedAllBeliefs;

%save('results.mat','handles');

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

function Messages = Propagate(padim,handles,allphivals,indsarr,Messages)
%%% Bundles together the passing functions, propagating messages throughout
%%% the image, simulating loopy propagation by treating the image at each
%%% step as a tree (a directed graph with no cliques)
psi = handles.Pipeline.Psi;
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

rows = size(padim,1)-2;
cols = size(padim,2)-2;
arr = zeros(rows,cols,3);
%%% repeats the peak matrices as necessary so they're the same size as x
c = repmat(handles.Pipeline.CellsPeak,1,cols);
n = repmat(handles.Pipeline.NucleiPeak,1,cols);
b = repmat(handles.Pipeline.BackgroundPeak,1,cols);
%%% for each row, for each column within that row, calculates the
%%% probability that a given pixel will be labeled in each of the three
%%% categories based on only its pixel intensity values.
for yind = 1:rows
    %%% x is the array of pixel values, [DNA;actin], for each corresponding
    %%% pixel in padim, accounting for the pad of zeros
    x = [padim(yind+1,2:end-1,1);padim(yind+1,2:end-1,2)];
    %%% Calculates the distance from each pixel's intensity value to the
    %%% peak of each pixel label for the whole image set.
    %%% Taking the phinorm of these vectors gives us a "hypotenuse" value
    %%% (so-called because of how it is derived) that represents the
    %%% distance, 2-pixel value-wise, that x is from each of these c, n, or
    %%% b expected values.  We then get the reciprocal of this so that
    %%% values are directly proportional to probability of a pixel being
    %%% correctly labeled as such.
    disB=(1./phinorm(b-x));
    disN=(1./phinorm(n-x));
    disC=(1./phinorm(c-x));
    %%% sums these values and normalize so that they sum to 1, making them a
    %%% legitimate description of probability.
    z=disB+disC+disN;
    probB=disB./z;
    probN=disN./z;
    probC=disC./z;
    %%% stores the results (which are an Nx3 array) into a 1xNx3 slice of
    %%% the results array
    arr(yind,:,:) = permute([probN; probC; probB],[3 2 1]);
end

function n=phinorm(x)
%%% see the notes in the comments in phi. gets a positive scalar (or
%%% vector, if input is 2 dimensional) value using, essentially, the
%%% pythagorean theorem if its parameter is a difference between two
%%% vectors
n=sqrt(sum(x.*x));

% function arr = phi2(padim,pObject)
% %%% This function takes the padded image that includes the nuclei (:,:,1)
% %%% and cell (:,:,2) staining images with a 1-pixel border of zeros, and an
% %%% expected percentage to be covered with objects in each image, and
% %%% returns an R-1xC-1x3 array (where input is RxCx2) containing the
% %%% probabilities of each label for each pixel.
% %%% For both cell and nucleus images, gets the background and object
% %%% gaussian probability curves
% if nargin == 1
%     pObject = 0.5;
% end
% [NDistBG,NDistObj] = MixtureOfGaussians2(padim(2:end-1,2:end-1,1),pObject);
% [CDistBG,CDistObj] = MixtureOfGaussians2(padim(2:end-1,2:end-1,2),pObject);
% arr = NDistObj;
% arr(:,:,2) = CDistObj;
% arr(:,:,3) = (NDistBG+CDistBG)/2; %%% IS THIS RIGHT?
% arr = arr./(repmat(sum(arr,3),[1 1 3]));
% 
% 
% function [BackgroundDistribution,ObjectDistribution] = MixtureOfGaussians2(OrigImage,pObject)
% %%% THIS VERSION RETURNS THE GAUSSIANS INSTEAD OF A THRESHOLD
% %%% This function finds a suitable threshold for the input image
% %%% Block. It assumes that the pixels in the image belong to either
% %%% a background class or an object class. 'pObject' is an initial guess
% %%% of the prior probability of an object pixel, or equivalently, the fraction
% %%% of the image that is covered by objects. Essentially, there are two steps.
% %%% First, a number of Gaussian distributions are estimated to match the
% %%% distribution of pixel intensities in OrigImage. Currently 3 Gaussian
% %%% distributions are fitted, one corresponding to a background class, one
% %%% corresponding to an object class, and one distribution for an intermediate
% %%% class. The distributions are fitted using the Expectation-Maximization (EM)
% %%% algorithm, a procedure referred to as Mixture of Gaussians modeling. When
% %%% the 3 Gaussian distributions have been fitted, it's decided whether the
% %%% intermediate class models background pixels or object pixels based on the
% %%% probability of an object pixel 'pObject' given by the user.
% Intensities = OrigImage(:);
% 
% %%% The number of classes is set to 2
% NumberOfClasses = 2;
% 
% %%% Transform the image into a vector. Also, if the image is (larger than 512x512),
% %%% select a subset of 512^2 pixels for speed. This should be enough to capture the
% %%% statistics in the image.
% %Intensities = OrigImage(:);
% %if length(Intensities) > 512^2
% %    indexes = randperm(length(Intensities));
% %    Intensities = Intensities(indexes(1:512^2));
% %end
% 
% %%% Get the probability for a background pixel
% pBackground = 1 - pObject;
% 
% %%% Initialize mean and standard deviations of the three Gaussian distributions
% %%% by looking at the pixel intensities in the original image and by considering
% %%% the percentage of the image that is covered by object pixels. Class 1 is the
% %%% background class and Class 3 is the object class. Class 2 is an intermediate
% %%% class and we will decide later if it encodes background or object pixels.
% %%% Also, for robustness the we remove 1% of the smallest and highest intensities
% %%% in case there are any quantization effects that have resulted in unaturally many
% %%% 0:s or 1:s in the image.
% Intensities = sort(Intensities);
% Intensities = Intensities(ceil(length(Intensities)*0.01):round(length(Intensities)*0.99));
% ClassMean(1) = Intensities(round(length(Intensities)*pBackground/2));                      %%% Initialize background class
% ClassMean(2) = Intensities(round(length(Intensities)*(1 - pObject/2)));                    %%% Initialize object class
% %ClassMean(2) = (ClassMean(1) + ClassMean(3))/2;                                            %%% Initialize intermediate class
% %%% Initialize standard deviations of the Gaussians. They should be the same to avoid problems.
% ClassStd(1:2) = 0.15;
% %%% Initialize prior probabilities of a pixel belonging to each class. The intermediate
% %%% class is gets some probability from the background and object classes.
% pClass(1) = 3/4*pBackground;
% %pClass(2) = 1/4*pBackground + 1/4*pObject;
% pClass(2) = 3/4*pObject;
% 
% %%% Apply transformation.  a < x < b, transform to log((x-a)/(b-x)).
% %a = - 0.000001;
% %b = 1.000001;
% %Intensities = log((Intensities-a)./(b-Intensities));
% %ClassMean = log((ClassMean-a)./(b - ClassMean))
% %ClassStd(1:3) = [1 1 1];
% 
% %%% Expectation-Maximization algorithm for fitting the three Gaussian distributions/classes
% %%% to the data. Note, the code below is general and works for any number of classes.
% %%% Iterate until parameters don't change anymore.
% delta = 1;
% while delta > 0.001
%     %%% Store old parameter values to monitor change
%     oldClassMean = ClassMean;
% 
%     %%% Update probabilities of a pixel belonging to the background or object1 or object2
%     for k = 1:NumberOfClasses
%         pPixelClass(:,k) = pClass(k)* 1/sqrt(2*pi*ClassStd(k)^2) * exp(-(Intensities - ClassMean(k)).^2/(2*ClassStd(k)^2));
%     end
%     pPixelClass = pPixelClass ./ repmat(sum(pPixelClass,2) + eps,[1 NumberOfClasses]);
% 
%     %%% Update parameters in Gaussian distributions
%     for k = 1:NumberOfClasses
%         pClass(k) = mean(pPixelClass(:,k));
%         ClassMean(k) = sum(pPixelClass(:,k).*Intensities)/(length(Intensities)*pClass(k));
%         ClassStd(k)  = sqrt(sum(pPixelClass(:,k).*(Intensities - ClassMean(k)).^2)/(length(Intensities)*pClass(k))) + sqrt(eps);    % Add sqrt(eps) to avoid division by zero
%     end
% 
%     %%% Calculate change
%     delta = sum(abs(ClassMean - oldClassMean));
% end
% 
% %%% Now the Gaussian distributions are fitted and we can describe the histogram of the pixel
% %%% intensities as the sum of these Gaussian distributions. To find a threshold we first have
% %%% to decide if the intermediate class 2 encodes background or object pixels. This is done by
% %%% choosing the combination of class probabilities 'pClass' that best matches the user input 'pObject'.
% Threshold = linspace(ClassMean(1),ClassMean(2),10000);
% Class1Gaussian = pClass(1) * 1/sqrt(2*pi*ClassStd(1)^2) * exp(-(Threshold - ClassMean(1)).^2/(2*ClassStd(1)^2));
% Class2Gaussian = pClass(2) * 1/sqrt(2*pi*ClassStd(2)^2) * exp(-(Threshold - ClassMean(2)).^2/(2*ClassStd(2)^2));
% %Class3Gaussian = pClass(3) * 1/sqrt(2*pi*ClassStd(3)^2) * exp(-(Threshold - ClassMean(3)).^2/(2*ClassStd(3)^2));
% figure, plot(Class1Gaussian);
% figure, plot(Class2Gaussian);
% %figure, plot(Class3Gaussian);
% %if abs(pClass(2) + pClass(3) - pObject) < abs(pClass(3) - pObject)
% %    %%% Intermediate class 2 encodes object pixels
% %    BackgroundDistribution = Class1Gaussian;
% %    ObjectDistribution = Class2Gaussian + Class3Gaussian;
% %else
% %    %%% Intermediate class 2 encodes background pixels
% %    BackgroundDistribution = Class1Gaussian + Class2Gaussian;
% %    ObjectDistribution = Class3Gaussian;
% %end
% 
% %%% Now, find the threshold at the intersection of the background distribution
% %%% and the object distribution.
% %[ignore,index] = min(abs(BackgroundDistribution - ObjectDistribution)); %#ok Ignore MLint
% %Threshold = Threshold(index);
% BackgroundDistribution = Class1Gaussian;
% ObjectDistribution = Class2Gaussian;