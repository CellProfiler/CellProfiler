function handles = ProbabilisticSegmentation(handles)

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
%   unlabeled pixel will first reach one of the primary objects.
%
%   For many images it is necessary to provide a seed for the background.
%   This module will identify a secondary object based on that seed.  To
%   remove this spurious object, use the FilterByObjectMeasurement module.
% 
%   See Grady's publications [doi:10.1109/TPAMI.2006.233, 
%   doi:10.1109/CVPR.2005.239] for details of the method.
%
% * Ljosa and Singh: XXX
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
if strcmp(char(handles.Settings.VariableValues{CurrentModuleNum,7}),'Yes')
    IdentChoiceList = {'Grady' 'Ljosa and Singh'};
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
%choiceVAR03 = Ljosa and Singh
%inputtypeVAR03 = popupmenu
OriginalIdentChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What did you call the images to be used to find the edges of the secondary objects?.
%infotypeVAR04 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = For Grady's algorithm, what is the weight parameter (beta)?
%defaultVAR05 = 90
Beta = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = For Ljosa and Singh's algorithm, what is the restart probability?
%defaultVAR06 = 0.001
restart_probability = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Do you want to run in test mode where each method for identifying secondary objects is compared?
%choiceVAR07 = No
%choiceVAR07 = Yes
TestMode = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

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
                    	 PrelimPrimaryLabelMatrixImage(indices), Beta);
    elseif strcmp(IdentChoice, 'Ljosa and Singh')
        nseeds = max(max(PrelimPrimaryLabelMatrixImage));
        SegmentationProbabilities = zeros(size(OrigImage,1), size(OrigImage,2), nseeds);
        for i = 1:nseeds
            indices = find(PrelimPrimaryLabelMatrixImage == i);
            x = fix(indices / size(OrigImage, 1));
            y = mod(indices, size(OrigImage, 1));
            seed = [x y];
            SegmentationProbabilities(:,:,i) = CPljosaprobseg(OrigImage, restart_probability, 1000, seed);
        end
        [highest_probability, FinalLabelMatrixImage] = max(SegmentationProbabilities, [], 3);
    else
      error 'Internal error: Unexpected IdentChoice'
    end
    EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);

    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);

    if strcmp(TestMode,'Yes')
        drawnow;
        %%% If the test mode window does not exist, it is created, but only
        %%% if it's at the starting image set (if the user closed the window
        %%% intentionally, we don't want to pop open a new one).
        SecondaryTestFigureNumber = findobj('Tag','ProbabilisticSegmentationTestModeFigure');
        if isempty(SecondaryTestFigureNumber) && handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet;
            %%% Creates the window, sets its tag, and puts some
            %%% text in it. The first lines are meant to find a suitable
            %%% figure number for the window, so we don't choose a
            %%% figure number that is being used by another module.
            %%% The integer 10 is arbitrary. Didn't want to
            %%% add 1 and 2 because other modules might be creating
            %%% a few windows.
            SecondaryTestFigureNumber = CPfigurehandle(handles)+10;
            CPfigure(handles,'Image',SecondaryTestFigureNumber);
            set(SecondaryTestFigureNumber,'Tag','ProbabilisticSegmentationTestModeFigure',...
                'name',['ProbabilisticSegmentation Test Display, cycle # ']);
            CPresizefigure(ObjectOutlinesOnOrigImage,'TwoByTwo',SecondaryTestFigureNumber);
        end
        %%% If the figure window DOES exist now, then calculate and display items
        %%% in it.
        if ~isempty(SecondaryTestFigureNumber)
            %%% Makes the figure window active.
            CPfigure(handles,'Image',SecondaryTestFigureNumber);
            %%% Updates the cycle number on the window.
            CPupdatefigurecycle(handles.Current.SetBeingAnalyzed,SecondaryTestFigureNumber);

            subplot(2,2,IdentChoiceNumber);
            CPimagesc(ObjectOutlinesOnOrigImage,handles);
            title(IdentChoiceList(IdentChoiceNumber));
        end
    end

    if strcmp(OriginalIdentChoice,IdentChoice)
        if ~isfield(handles.Measurements,SecondaryObjectName)
            handles.Measurements.(SecondaryObjectName) = {};
        end

        if ~isfield(handles.Measurements,PrimaryObjectName)
            handles.Measurements.(PrimaryObjectName) = {};
        end

        %        handles = CPrelateobjects(handles,SecondaryObjectName,PrimaryObjectName,FinalLabelMatrixImage,EditedPrimaryLabelMatrixImage,ModuleName);

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

    end
end