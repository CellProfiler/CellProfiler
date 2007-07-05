function handles = ProbSeg(handles)

% Help for the Probabilistic Segmentation module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies objects (e.g., neurons) using "seed" objects identified by
% an Identify Primary module (e.g., nuclei).
% *************************************************************************
%
% This module identifies secondary objects (e.g., neurons) based on two
% inputs: (1) a previous module's identification of primary objects (e.g.,
% nuclei) and (2) an image stained for the secondary objects. Each primary
% object is assumed to be completely within a secondary object (e.g.,
% nuclei are completely within cells stained for actin).
%
% Settings:
%
% Methods to identify secondary objects:
% * Grady: Given predefined primary objects (seeds), the algorithm
%   determines the probability that a random walker starting at each
%   unlabeled pixel will first reach one of the primary objects.  One of
%   the seeds should be the background.
%   [doi:10.1109/TPAMI.2006.233, doi:10.1109/CVPR.2005.239]
%
% * Ljosa: XXX
%   [doi:10.1109/ICDM.2006.129]
%
% Note: Primary identify modules produce two (hidden) output images that
% are used by this module. The Segmented image contains the final, edited
% primary objects (i.e. objects at the border and those that are too small
% or large have been excluded). The SmallRemovedSegmented image is the
% same except that the objects at the border and the large objects have
% been included. These extra objects are used to perform the identification
% of secondary object outlines, since they are probably real objects (even
% if we don't want to measure them). Small objects are not used at this
% stage because they are more likely to be artifactual, and so they
% therefore should not "claim" any secondary object pixels.
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Broad Institute
% Copyright 2007
% Website: http://www.cellprofiler.org
%
% $Revision: 1809 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%%% Sets up loop for test mode.
if strcmp(char(handles.Settings.VariableValues{CurrentModuleNum,5}),'Yes')
    IdentChoiceList = {'Grady' 'Ljosa'};
else
    IdentChoiceList = {char(handles.Settings.VariableValues{CurrentModuleNum,3})};
end

%textVAR01 = What did you call the primary objects you want to create secondary objects around?
%infotypeVAR01 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Select the method to identify the secondary objects:
%choiceVAR03 = Grady
%choiceVAR03 = Ljosa
%inputtypeVAR03 = popupmenu
OriginalIdentChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What did you call the images to be used to find the edges of the secondary objects?.
%infotypeVAR04 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

% XXX: Add variables for restart probability, momentum, and distance
% on/off.

%textVAR05 = Do you want to run in test mode where each method for identifying secondary objects is compared?
%choiceVAR05 = No
%choiceVAR05 = Yes
TestMode = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
PrelimPrimaryLabelMatrixImage = CPretrieveimage(handles,['SmallRemovedSegmented', PrimaryObjectName],ModuleName,'DontCheckColor','DontCheckScale',size(OrigImage));

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
EditedPrimaryLabelMatrixImage = CPretrieveimage(handles,['Segmented', PrimaryObjectName],ModuleName,'DontCheckColor','DontCheckScale',size(OrigImage));

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

for IdentChoiceNumber = 1:length(IdentChoiceList)
    IdentChoice = IdentChoiceList{IdentChoiceNumber};
    if strcmp(IdentChoice,'Grady')
      %%% Leo Grady's random-walk--based segmentation method.
      indices = find(PrelimPrimaryLabelMatrixImage > 0);
      [FinalLabelMatrixImage, SegmentationProbabilities] = ...
          CPrandomwalker(OrigImage, indices, ...
                    	 PrelimPrimaryLabelMatrixImage(indices));
      EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);
      % 
    else
      error 'Internal error: Unexpected IdentChoice'
    end

    %%% Calculates the ColoredLabelMatrixImage, which will be used to
    %%% visualize the probabilistic segmentation in subplot(1,2,2).
    global ColoredLabelMatrixImage;
    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);

    if strcmp(TestMode,'Yes')
        SecondaryTestFig = findobj('Tag','SecondaryTestFigure');
        if isempty(SecondaryTestFig)
            SecondaryTestFig = CPfigure(handles,'Image','Tag','SecondaryTestFigure','Name','Secondary Test Figure');
        else
            CPfigure(handles,'Image',SecondaryTestFig);
        end
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(ObjectOutlinesOnOrigImage,'TwoByTwo',SecondaryTestFig);
        end
        subplot(2,2,IdentChoiceNumber);
        CPimagesc(ObjectOutlinesOnOrigImage,handles);
        title(IdentChoiceList(IdentChoiceNumber));
    end

    if strcmp(OriginalIdentChoice,IdentChoice)
        if ~isfield(handles.Measurements,SecondaryObjectName)
            handles.Measurements.(SecondaryObjectName) = {};
        end

        if ~isfield(handles.Measurements,PrimaryObjectName)
            handles.Measurements.(PrimaryObjectName) = {};
        end

        handles = CPrelateobjects(handles,SecondaryObjectName,PrimaryObjectName,FinalLabelMatrixImage,EditedPrimaryLabelMatrixImage,ModuleName);

        %%%%%%%%%%%%%%%%%%%%%%%
        %%% DISPLAY RESULTS %%%
        %%%%%%%%%%%%%%%%%%%%%%%
        drawnow

        ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
        if any(findobj == ThisModuleFigureNumber)
            %%% Activates the appropriate figure window.
            CPfigure(handles,'Image',ThisModuleFigureNumber);
            if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
                CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber);
            end
            %%% A subplot of the figure window is set to display the original image.
            subplot(2,1,1);
            CPimagesc(OrigImage,handles);
            title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
            %%% A subplot of the figure window is set to display the
            %%% probabilistic segmentation.
            subplot(2,1,2);
            CPimagesc(CPvisualizeProbabilisticSegmentation(ColoredLabelMatrixImage, FinalLabelMatrixImage, SegmentationProbabilities), handles);
            title(['Probabilistic segmentation of ', SecondaryObjectName]);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% SAVE DATA TO HANDLES STRUCTURE %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        drawnow

        %%% Saves the final, segmented label matrix image of secondary objects to
        %%% the handles structure so it can be used by subsequent modules.
        fieldname = ['Segmented',SecondaryObjectName];
        handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

        handles.Pipeline.(['SegmentationProbabilities', SecondaryObjectName]) = SegmentationProbabilities;

        %%% Saves the ObjectCount, i.e. the number of segmented objects.
        if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
            handles.Measurements.Image.ObjectCountFeatures = {};
            handles.Measurements.Image.ObjectCount = {};
        end
        column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,SecondaryObjectName)));
        if isempty(column)
            handles.Measurements.Image.ObjectCountFeatures(end+1) = {SecondaryObjectName};
            column = length(handles.Measurements.Image.ObjectCountFeatures);
        end
        handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

        %%% Saves the location of each segmented object
        handles.Measurements.(SecondaryObjectName).LocationFeatures = {'CenterX','CenterY'};
        tmp = regionprops(FinalLabelMatrixImage,'Centroid');
        Centroid = cat(1,tmp.Centroid);
        if isempty(Centroid)
            Centroid = [0 0];
        end
        handles.Measurements.(SecondaryObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};
    end
end