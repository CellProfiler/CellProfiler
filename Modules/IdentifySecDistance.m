function handles = IdentifySecDistance(handles)

% Help for the Identify Secondary Distance module:
% Category: Object Processing
%
% Based on another module's identification of primary objects, this
% module identifies secondary objects when no specific staining is
% available.  The edges of the primary objects are simply expanded a
% particular distance to create the secondary objects. For example, if
% nuclei are labeled but there is no stain to help locate cell edges,
% the nuclei can simply be expanded in order to estimate the cell's
% location.  This is a standard module used in commercial software and
% is known as the 'donut' or 'annulus' approach for identifying the
% cytoplasmic compartment.
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module, this
% module produces a grayscale image where each object is a different
% intensity, which you can save using the Save Images module using the
% name: Segmented + whatever you called the objects (e.g.
% SegmentedCells).
%
%    Additional image(s) are calculated by this module and can be 
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% See also IDENTIFYSECPROPAGATE, IDENTIFYSECWATERSHED.

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




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the primary objects you want to create secondary objects around?
%infotypeVAR01 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the secondary objects identified by this module?
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = On which image would you like to display the outlines of the secondary objects?
%infotypeVAR03 = imagegroup
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Set the number of pixels by which to expand the primary objects [Positive number]
%defaultVAR04 = 10
DistanceToDilate = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = What do you want to call the image of the outlines of the objects?
%choiceVAR05 = Do not save
%choiceVAR05 = OutlinedNuclei
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,5}); 
%inputtypeVAR05 = popupmenu custom

%textVAR06 =  What do you want to call the labeled matrix image?
%choiceVAR06 = Do not save
%choiceVAR06 = LabeledNuclei
%infotypeVAR06 = imagegroup indep
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,6}); 
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR07 = RGB
%choiceVAR07 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,7}); 
%inputtypeVAR07 = popupmenu

%%%VariableRevisionNumber = 01

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', OrigImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Distance module could not find the input image.  It was supposed to be named ', OrigImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used for dilation. Checks first to see
%%% whether the appropriate image exists.
fieldname = ['Segmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Distance module, you must have previously run a module that generates an image with the primary objects identified.  You specified in the Identify Secondary Distance module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Distance module cannot locate this image.']);
end
PrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
fieldname = ['SmallRemovedSegmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Distance module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Distance module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Distance module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Creates the structuring element using the user-specified size.
StructuringElement = strel('disk', DistanceToDilate);
%%% Dilates the preliminary label matrix image (edited for small only).
DilatedPrelimSecObjectLabelMatrixImage = imdilate(PrelimPrimaryLabelMatrixImage, StructuringElement);
%%% Converts to binary.
DilatedPrelimSecObjectBinaryImage = im2bw(DilatedPrelimSecObjectLabelMatrixImage,.5);
%%% Computes nearest neighbor image of nuclei centers so that the dividing
%%% line between secondary objects is halfway between them rather than
%%% favoring the primary object with the greater label number.
[ignore, Labels] = bwdist(full(PrelimPrimaryLabelMatrixImage>0)); %#ok We want to ignore MLint error checking for this line.
drawnow
%%% Remaps labels in Labels to labels in PrelimPrimaryLabelMatrixImage.
ExpandedRelabeledDilatedPrelimSecObjectImage = PrelimPrimaryLabelMatrixImage(Labels);
%%% Removes the background pixels (those not labeled as foreground in the
%%% DilatedPrelimSecObjectBinaryImage). This is necessary because the
%%% nearest neighbor function assigns *every* pixel to a nucleus, not just
%%% the pixels that are part of a secondary object.
RelabeledDilatedPrelimSecObjectImage = zeros(size(ExpandedRelabeledDilatedPrelimSecObjectImage));
RelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage) = ExpandedRelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage);
drawnow
%%% Removes objects that are not in the edited PrimaryLabelMatrixImage.
LookUpTable = sortrows(unique([PrelimPrimaryLabelMatrixImage(:) PrimaryLabelMatrixImage(:)],'rows'),1);
LookUpColumn = LookUpTable(:,2);
FinalSecObjectsLabelMatrixImage = LookUpColumn(RelabeledDilatedPrelimSecObjectImage+1);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalSecObjectsLabelMatrixImage)) >= 1
        cmap = jet(max(64,max(FinalSecObjectsLabelMatrixImage(:))));
        ColoredLabelMatrixImage = label2rgb(FinalSecObjectsLabelMatrixImage, cmap, 'k', 'shuffle');
    else ColoredLabelMatrixImage = FinalSecObjectsLabelMatrixImage;
    end
    %%% Calculates ObjectOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,3).
    StructuringElement3 = [0 0 0; 0 1 -1; 0 0 0];
    OutlinesDirection1 = filter2(StructuringElement3, FinalSecObjectsLabelMatrixImage);
    OutlinesDirection2 = filter2(StructuringElement3', FinalSecObjectsLabelMatrixImage);
    SecondaryObjectOutlines = OutlinesDirection1 | OutlinesDirection2;
    %%% Overlay the watershed lines on the original image.
    ObjectOutlinesOnOrigImage = OrigImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    ObjectOutlinesOnOrigImage(SecondaryObjectOutlines == 1) = LineIntensity;
    %%% Calculates BothOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,4).
    %%% Converts the PrimaryLabelMatrixImage to binary.
    PrimaryBinaryImage = im2bw(PrimaryLabelMatrixImage,.5);
    %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
    StructuringElement2 = strel('square',3);
    DilatedPrimaryBinaryImage = imdilate(PrimaryBinaryImage, StructuringElement2);
    %%% Subtracts the PrimaryBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedPrimaryBinaryImage - PrimaryBinaryImage;
    %%% Writes the outlines onto the original image.
    BothOutlinesOnOrigImage = ObjectOutlinesOnOrigImage;
    BothOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);CPcolormap(handles);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(FinalSecObjectsLabelMatrixImage); CPcolormap(handles);title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with outlines drawn on top.
    subplot(2,2,3); imagesc(ObjectOutlinesOnOrigImage); CPcolormap(handles); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original image
    %%% with outlines drawn on top.
    subplot(2,2,4); imagesc(BothOutlinesOnOrigImage); CPcolormap(handles); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
    CPFixAspectRatio(OrigImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent modules.
fieldname = ['Segmented',SecondaryObjectName];
handles.Pipeline.(fieldname) = FinalSecObjectsLabelMatrixImage;

%%% Saves the ObjectCount, i.e. the number of segmented objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,SecondaryObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' SecondaryObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalSecObjectsLabelMatrixImage(:));


%%% Saves the location of each segmented object
handles.Measurements.(SecondaryObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalSecObjectsLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(SecondaryObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
try
    if ~strcmp(SaveColored,'Do not save')
        if strcmp(SaveMode,'RGB')
            handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
        else
            handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
        end
    end
    if ~strcmp(SaveOutlined,'Do not save')
        handles.Pipeline.(SaveOutlined) = ObjectOutlinesOnOrigImage;
    end
catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
end
