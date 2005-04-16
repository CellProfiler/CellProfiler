function handles = MeasureTexture(handles)

% Help for the Measure Texture module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts texture features of each
% object based on a corresponding grayscale image. Measurements are
% recorded for each object.
%
% How it works:
% Retrieves a segmented image, in label matrix format, and a
% corresponding original grayscale image and makes measurements of the
% objects that are segmented in the image. The label matrix image should
% be "compacted": that is, each number should correspond to an object,
% with no numbers skipped.
% So, if some objects were discarded from the label matrix image, the
% image should be converted to binary and re-made into a label matrix
% image before feeding into this module.
%
% See also MEASUREAREAOCCUPIED,
% MEASUREAREASHAPECOUNTLOCATION,
% MEASURECORRELATION,
% MEASURETOTALINTENSITY.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.
%
% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.


%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT:
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the module.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this module with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown.
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the greyscale images you want to measure?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%textVAR02 = What did you call the segmented objects that you want to measure?
%defaultVAR02 = Nuclei
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%textVAR03 = Type / in unused boxes.
%defaultVAR03 = Cells
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%textVAR04 =
%defaultVAR04 = /
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 02

%%% Set up the window for displaying the results
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);
    figure(ThisModuleFigureNumber);
    set(ThisModuleFigureNumber,'color',[1 1 1])
    columns = 1;
end


%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:3
    ObjectName = ObjectNameList{i};
    if strcmp(ObjectName,'/') == 1
        break
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Reads (opens) the image you want to analyze and assigns it to a variable,
    %%% "OrigImageToBeAnalyzed".
    fieldname = ['', ImageName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Texture module, you must have previously run a module that loads a greyscale image.  You specified in the MeasureTexture module that the desired image was named ', ImageName, ' which should have produced an image in the handles structure called ', fieldname, '. The Measure Texture module cannot locate this image.']);
    end
    OrigImageToBeAnalyzed = handles.Pipeline.(fieldname);


    %%% Checks that the original image is two-dimensional (i.e. not a color
    %%% image), which would disrupt several of the image functions.
    if ndims(OrigImageToBeAnalyzed) ~= 2
        error('Image processing was canceled because the Measure Texture module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
    end

    %%% Retrieves the label matrix image that contains the segmented objects which
    %%% will be measured with this module.
    fieldname = ['Segmented', ObjectName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Texture module, you must have previously run a module that generates an image with the objects identified.  You specified in the Measure Texture module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Measure Texture module cannot locate this image.']);
    end
    LabelMatrixImage = handles.Pipeline.(fieldname);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    % PROGRAMMING NOTE
    % HANDLES STRUCTURE:
    %       In CellProfiler (and Matlab in general), each independent
    % function (module) has its own workspace and is not able to 'see'
    % variables produced by other modules. For data or images to be shared
    % from one module to the next, they must be saved to what is called
    % the 'handles structure'. This is a variable, whose class is
    % 'structure', and whose name is handles. The contents of the handles
    % structure are printed out at the command line of Matlab using the
    % Tech Diagnosis button. The only variables present in the main
    % handles structure are handles to figures and gui elements.
    % Everything else should be saved in one of the following
    % substructures:
    %
    % handles.Settings:
    %       Everything in handles.Settings is stored when the user uses
    % the Save pipeline button, and these data are loaded into
    % CellProfiler when the user uses the Load pipeline button. This
    % substructure contains all necessary information to re-create a
    % pipeline, including which modules were used (including variable
    % revision numbers), their setting (variables), and the pixel size.
    %   Fields currently in handles.Settings: PixelSize, ModuleNames,
    % VariableValues, NumbersOfVariables, VariableRevisionNumbers.
    %
    % handles.Pipeline:
    %       This substructure is deleted at the beginning of the
    % analysis run (see 'Which substructures are deleted prior to an
    % analysis run?' below). handles.Pipeline is for storing data which
    % must be retrieved by other modules. This data can be overwritten as
    % each image set is processed, or it can be generated once and then
    % retrieved during every subsequent image set's processing, or it can
    % be saved for each image set by saving it according to which image
    % set is being analyzed, depending on how it will be used by other
    % modules. Any module which produces or passes on an image needs to
    % also pass along the original filename of the image, named after the
    % new image name, so that if the SaveImages module attempts to save
    % the resulting image, it can be named by appending text to the
    % original file name.
    %   Example fields in handles.Pipeline: FileListOrigBlue,
    % PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which contains the actual image).
    %
    % handles.Current:
    %       This substructure contains information needed for the main
    % CellProfiler window display and for the various modules to
    % function. It does not contain any module-specific data (which is in
    % handles.Pipeline).
    %   Example fields in handles.Current: NumberOfModules,
    % StartupDirectory, DefaultOutputDirectory, DefaultImageDirectory,
    % FilenamesInImageDir, CellProfilerPathname, ImageToolHelp,
    % DataToolHelp, FigureNumberForModule01, NumberOfImageSets,
    % SetBeingAnalyzed, TimeStarted, CurrentModuleNumber.
    %
    % handles.Preferences:
    %       Everything in handles.Preferences is stored in the file
    % CellProfilerPreferences.mat when the user uses the Set Preferences
    % button. These preferences are loaded upon launching CellProfiler.
    % The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
    % fields can be changed for the current session by the user using edit
    % boxes in the main CellProfiler window, which changes their values in
    % handles.Current. Therefore, handles.Current is most likely where you
    % should retrieve this information if needed within a module.
    %   Fields currently in handles.Preferences: PixelSize, FontSize,
    % DefaultModuleDirectory, DefaultOutputDirectory,
    % DefaultImageDirectory.
    %
    % handles.Measurements:
    %       Everything in handles.Measurements contains data specific to each
    % image set analyzed for exporting. It is used by the ExportMeanImage
    % and ExportCellByCell data tools. This substructure is deleted at the
    % beginning of the analysis run (see 'Which substructures are deleted
    % prior to an analysis run?' below).
    %    Note that two types of measurements are typically made: Object
    % and Image measurements.  Object measurements have one number for
    % every object in the image (e.g. ObjectArea) and image measurements
    % have one number for the entire image, which could come from one
    % measurement from the entire image (e.g. ImageTotalIntensity), or
    % which could be an aggregate measurement based on individual object
    % measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
    % ensure that your data will be extracted properly. It is likely that
    % Subobject will become a new prefix, when measurements will be
    % collected for objects contained within other objects.
    %       Saving measurements: The data extraction functions of
    % CellProfiler are designed to deal with only one "column" of data per
    % named measurement field. So, for example, instead of creating a
    % field of XY locations stored in pairs, they should be split into a
    % field of X locations and a field of Y locations. It is wise to
    % include the user's input for 'ObjectName' or 'ImageName' as part of
    % the fieldname in the handles structure so that multiple modules can
    % be run and their data will not overwrite each other.
    %   Example fields in handles.Measurements: ImageCountNuclei,
    % ObjectAreaCytoplasm, FilenameOrigBlue, PathnameOrigBlue,
    % TimeElapsed.
    %
    % Which substructures are deleted prior to an analysis run?
    %       Anything stored in handles.Measurements or handles.Pipeline
    % will be deleted at the beginning of the analysis run, whereas
    % anything stored in handles.Settings, handles.Preferences, and
    % handles.Current will be retained from one analysis to the next. It
    % is important to think about which of these data should be deleted at
    % the end of an analysis run because of the way Matlab saves
    % variables: For example, a user might process 12 image sets of nuclei
    % which results in a set of 12 measurements ("ImageTotalNucArea")
    % stored in handles.Measurements. In addition, a processed image of
    % nuclei from the last image set is left in the handles structure
    % ("SegmNucImg"). Now, if the user uses a different module which
    % happens to have the same measurement output name "ImageTotalNucArea"
    % to analyze 4 image sets, the 4 measurements will overwrite the first
    % 4 measurements of the previous analysis, but the remaining 8
    % measurements will still be present. So, the user will end up with 12
    % measurements from the 4 sets. Another potential problem is that if,
    % in the second analysis run, the user runs only a module which
    % depends on the output "SegmNucImg" but does not run a module that
    % produces an image by that name, the module will run just fine: it
    % will just repeatedly use the processed image of nuclei leftover from
    % the last image set, which was left in handles.Pipeline.

  
    %%% Initilize measurement structure
    Haralick = [];
    HaralickFeatures = {'H1. AngularSecondMoment',...
        'H2. Contrast',...
        'H3. Correlation',...
        'H4. Variance',...
        'H5. InverseDifferenceMoment',...
        'H6. SumAverage',...
        'H7. SumVariance',...
        'H8. SumEntropy',...
        'H9. Entropy',...
        'H10. DifferenceVariance',...
        'H11. DifferenceEntropy',...
        'H12. InformationMeasure1',...
        'H13. InformationMeasure2'};

    Gabor = [];
    GaborFeatures    = {'Gabor1x',...
        'Gabor1y',...
        'Gabor2x',...
        'Gabor2y',...
        'Gabor3x',...
        'Gabor3y'};
    %%% Count objects
    ObjectCount = max(LabelMatrixImage(:));

    if ObjectCount > 0
        
        %%% Get Gabor features.
        %%% The Gabor features are calculated by convolving the entire
        %%% image with Gabor filters and then extracting the filter output
        %%% value in the centroids of the objects in LabelMatrixImage
        
        % Adjust size of filter to size of objects in the image
        % The centroids indicate where we should measure the Gabor
        % filter output
        tmp = regionprops(LabelMatrixImage,'Area','Centroid');
        MedianArea = median(cat(1,tmp.Area));
        sigma = sqrt(MedianArea/pi);
        
        % Round centroids and find linear index for them.
        % The centroids are stored in [column,row] order.
        Centroids = round(cat(1,tmp.Centroid));
        Centroidsindex = sub2ind(size(LabelMatrixImage),Centroids(:,2),Centroids(:,1));
        
        % Use Gabor filters with three different frequencies
        f = [0.06 0.12 0.24];

        % Angle direction, filter along the x-axis and y-axis
        theta = [0 pi/2];

        % Create kernel coordinates
        KernelSize = round(2*sigma);
        [x,y]=meshgrid(-KernelSize:KernelSize,-KernelSize:KernelSize);
     
        % Apply Gabor filters and store filter outputs in the Centroid pixels
        Fourier_OrigImageToBeAnalyzed = fft2(OrigImageToBeAnalyzed);
        GaborFeatureNo = 1;
        Gabor = zeros(ObjectCount,length(f)*length(theta));                              % Initialize measurement matrix
        for m = 1:length(f)
            for n = 1:length(theta)
                
                % Calculate Gabor filter kernel 
                % Scale by 1000 to get measurements in a convenient range
                g = 1000*1/(2*pi*sigma^2)*exp(-(x.^2 + y.^2)/(2*sigma^2)).*exp(2*pi*sqrt(-1)*f(m)*(x*cos(theta(n))+y*sin(theta(n))));
                
                
                % Perform filtering in the Fourier domain
                q = ifft2(fft2(g,size(OrigImageToBeAnalyzed,1),size(OrigImageToBeAnalyzed,2)).*Fourier_OrigImageToBeAnalyzed);
              
                % Store filter output
                Gabor(:,GaborFeatureNo) = abs(q(Centroidsindex));
                GaborFeatureNo = GaborFeatureNo + 1;
            
            end
        end

        %%% Get Haralick features.
        %%% Have to loop over the objects
        Haralick = zeros(ObjectCount,13);
        [sr sc] = size(LabelMatrixImage);
        for Object = 1:ObjectCount

            %%% Cut patch so that we don't have to deal with entire image
            [r,c] = find(LabelMatrixImage == Object);
            rmax = min(sr,max(r));
            rmin = max(1,min(r));
            cmax = min(sc,max(c));
            cmin = max(1,min(c));
            BWim   = LabelMatrixImage(rmin:rmax,cmin:cmax) == Object;
            Greyim = OrigImageToBeAnalyzed(rmin:rmax,cmin:cmax);

            %%% Get Haralick features
            Haralick(Object,:) = CalculateHaralick(Greyim,BWim);
        end
    end
    %%% Save measurements
    handles.Measurements.(ObjectName).(['Texture_',ImageName,'Features']) = cat(2,HaralickFeatures,GaborFeatures);
    handles.Measurements.(ObjectName).(['Texture_',ImageName])(handles.Current.SetBeingAnalyzed) = {[Haralick Gabor]};


    %%% Report measurements
    FontSize = get(0,'UserData');

    if any(findobj == ThisModuleFigureNumber);
        % This first block writes the same text several times
        % Header
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf('Average texture features for image set #%d',handles.Current.SetBeingAnalyzed));

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

        % Text for Gabor features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Gabor features:');
        for k = 1:6
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',FontSize,'string',GaborFeatures{k});
        end

        % Text for Haralick features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.5 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Haralick features:');
        for k = 1:10
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.5-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',FontSize,'string',HaralickFeatures{k});
        end

        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.9 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.85 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'string',num2str(ObjectCount));

        if ObjectCount > 0
            % Gabor features
            for k = 1:6
                q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.8-0.04*k 0.2 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Gabor(:,k))));
            end

            % Haralick features
            for k = 1:10
                q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.5-0.04*k 0.2 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Haralick(:,k))));
            end
        end
        % This variable is used to write results in the correct column
        % and to determine the correct window size
        columns = columns + 1;
    end
end
drawnow

function H = CalculateHaralick(im,mask,area)
%
% This function calculates so-called Haralick features, which are
% based on the co-occurence matrix. The function takes two inputs:
%
% im    - A grey level image
% mask  - A binary mask
%
% Currently, the implementation uses 8 different grey levels
% and calculates the co-occurence matrix for a horizontal shift
% of 1 pixel.
%
% The original reference is:
% Haralick et al. (1973)
% Textural Features for Image Classification.
% IEEE Transaction on Systems
% Man, Cybernetics, SMC-3(6):610-621.
%
% BEWARE: There are lots of erroneous formulas for the Haralick features in
% the literature. There is also an error in the original paper.
%

% Number of greylevels to use
Levels = 8;

% Quantize the image into a lower number
% of grey levels (specified by Levels)
BinEdges = linspace(0,1,Levels+1);
im = im - min(im(:));
im = im/max(im(:));
qim = zeros(size(im));
for k = 1:Levels
    qim(find(im > BinEdges(k))) = k;
end

% Shift 1 step to the right
im1 = qim(:,1:end-1); im1 = im1(:);
im2 = qim(:,2:end);   im2 = im2(:);

% Remove cases where at least one position is
% outside the mask.
m1 = mask(:,1:end-1); m1 = m1(:);
m2 = mask(:,2:end);   m2 = m2(:);
index = (sum([m1 m2],2) == 2);
im1 = im1(index);
im2 = im2(index);

%%% Calculate co-occurence matrix
P = zeros(Levels);
for k = 1:Levels
    index = find(im1==k);
    if ~isempty(index)
        P(k,:) = hist(im2(index),[1:Levels]);
    else
        P(k,:) = zeros(1,Levels);
    end
end
P = P/length(im1);


%%% Calculate features from the co-occurence matrix
% First, pre-calculate a few quantities that are used in
% several features.
px = sum(P,2);
py = sum(P,1);
mux = sum([1:Levels]'.*px);
muy = sum([1:Levels].*py);
sigmax = sqrt(sum(([1:Levels]' - mux).^2.*px));
sigmay = sqrt(sum(([1:Levels] - muy).^2.*py));
HX = -sum(px.*log(px+eps));
HY = -sum(py.*log(py+eps));
HXY = -sum(P(:).*log(P(:)+eps));
HXY1 = -sum(sum(P.*log(px*py+eps)));
HXY2 = -sum(sum(px*py .* log(px*py+eps)));

p_xplusy = zeros(2*Levels-1,1);      % Range 2:2*Levels
p_xminusy = zeros(Levels,1);         % Range 0:Levels-1
for x=1:Levels
    for y = 1:Levels
        p_xplusy(x+y-1) = p_xplusy(x+y-1) + P(x,y);
        p_xminusy(abs(x-y)+1) = p_xminusy(abs(x-y)+1) + P(x,y);
    end
end

% H1. Angular Second Moment
H1 = sum(P(:).^2);

% H2. Contrast
H2 = sum([0:Levels-1]'.^2.*p_xminusy);

% H3. Correlation
H3 = (sum(sum([1:Levels]'*[1:Levels].*P)) - mux*muy)/(sigmax*sigmay);

% H4. Sum of Squares: Variation
H4 = sigmax^2;

% H5. Inverse Difference Moment
H5 = sum(sum(1./(1+toeplitz(0:Levels-1).^2).*P));

% H6. Sum Average
H6 = sum([2:2*Levels]'.*p_xplusy);

% H7. Sum Variance (error in Haralick's original paper here)
H7 = sum(([2:2*Levels]' - H6).^2 .* p_xplusy);

% H8. Sum Entropy
H8 = -sum(p_xplusy .* log(p_xplusy+eps));

% H9. Entropy
H9 = - sum(P(:).*log(P(:)+eps));

% H10. Difference Variance
H10 = sum(p_xminusy.*([0:Levels-1]' - sum([0:Levels-1]'.*p_xminusy)).^2);

% H11. Difference Entropy
H11 = - sum(p_xminusy.*log(p_xminusy+eps));

% H12. Information Measure of Correlation 1
H12 = (HXY-HXY1)/max(HX,HY);

% H13. Information Measure of Correlation 2
H13 = real(sqrt(1-exp(-2*(HXY2-HXY))));             % An imaginary result has been encountered once, reason unclear

% H14. Max correlation coefficient (not currently used)
% Q = zeros(Levels);
% for i = 1:Levels
%     for j = 1:Levels
%         Q(i,j) = sum(P(i,:).*P(j,:)/(px(i)*py(j)));
%     end
% end
% [V,lambda] = eig(Q);
% lambda = sort(diag(lambda));
% H14 = sqrt(max(0,lambda(end-1)));

H = [H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13];






% % This function calculates Gabor features in a different way
% % It may be better but it's also considerably slower. 
% % It's called by Gabor(Object,:) = CalculateGabor(Greyim,BWim,sigma);
% function G = CalculateGabor(im,mask,sigma,flag)
% %
% % This function calculates Gabor features, which measure
% % the energy in different frequency sub-bands. The Gabor
% % transform is essentially equivalent to a wavelet transform.
% %
% % im    - A grey level image
% % mask  - A binary mask
% % sigma - Scale parameter for the Gaussian weight function
% 
% % Use Gabor filters with three different frequencies
% f = [0.06 0.12 0.24];
% 
% % Filter along the x-axis and y-axis
% theta = [0 pi/2];
% 
% % Match the filter kernel size to the input patch size
% [sr,sc] = size(mask);
% if rem(sr,2) == 0,ty = [-sr/2:sr/2-1];else ty = [-(sr-1)/2:(sr-1)/2];end
% if rem(sc,2) == 0,tx = [-sc/2:sc/2-1];else tx = [-(sc-1)/2:(sc-1)/2];end
% [x,y]=meshgrid(tx,ty);
% 
% % Calculate the Gabor features
% G = zeros(length(theta),length(f));
% for m = 1:length(f)
%     for n = 1:length(theta)
%         
%         % Calculate Gabor filter kernel
%         g = 1/(2*pi*sigma^2)*exp(-(x.^2 + y.^2)/(2*sigma^2)).*exp(2*pi*sqrt(-1)*f(m)*(x*cos(theta(n))+y*sin(theta(n))));
% 
%         % Use Normalized Convolution to calculate filter responses. This
%         % method only include object pixels for calculating the filter
%         % response and excludes surrounding background pixels.
%         % See Farneback, 2002. "Polynomial Expansion for Orientation and
%         % Motion Estimation". PhD Thesis
%         gr = real(g);
%         gi = imag(g);
%         B = [gr(:) gi(:)];
%         Wc = diag(mask(:));
%         r = inv(B'*Wc*B)*B'*Wc*im(:); 
%         G(n,m) = sqrt(sum(r.^2));
%         
%         % Direct way of calculating filter responses
%         %tmpr = sum(sum(real(g).*im));
%         %tmpi = sum(sum(imag(g).*im));
%         %G(n,m) = sqrt(tmpr.^2+tmpi.^2);
%     end
% end
% G = G(:)';
% 
