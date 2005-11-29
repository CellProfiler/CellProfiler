function handles = IdentifyTertiarySubregion(handles)

% Help for the Identify Tertiary Subregion module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies 3rd order obects (e.g. cytoplasm) by removing the 1st order
% objects (e.g. nuclei) from 2nd order objects (e.g. cells) leaving a
% doughnut shape.
% *************************************************************************
%
% This module will take the identified objects specified in the first
% box and remove from them the identified objects specified in the
% second box. For example, "subtracting" the nuclei from the cells
% will leave just the cytoplasm, the properties of which can then be
% measured by Measure modules. The first objects should therefore be
% equal in size or larger than the second objects and must completely
% contain the second objects.  Both inputs should be objects produced by
% identify modules, not images. Note that creating
% subregions using this module can result in objects that are not
% contiguous, which does not cause problems when running the Measure
% Intensity and Texture module, but does cause problems when running
% the Measure Area Shape Intensity Texture module because calculations
% of the perimeter, aspect ratio, solidity, etc. cannot be made for
% noncontiguous objects.
%
% See also identify Primary and Identify Secondary modules.

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the larger identified objects?
%infotypeVAR01 = objectgroup
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the smaller identified objects?
%infotypeVAR02 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the new subregions?
%defaultVAR03 = Cytoplasm
%infotypeVAR03 = objectgroup indep
SubregionObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR04 = Do not save
%infotypeVAR04 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
fieldname = ['Segmented', PrimaryObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', PrimaryObjectName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
PrimaryObjectImage = handles.Pipeline.(fieldname);

%%% Retrieves the Secondary object segmented image.
fieldname = ['Segmented', SecondaryObjectName];
if isfield(handles.Pipeline, fieldname) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', SecondaryObjectName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
SecondaryObjectImage = handles.Pipeline.(fieldname);

%%% Checks that these images are two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(PrimaryObjectImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end
if ndims(SecondaryObjectImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Erodes the primary object image and then subtracts it from the
%%% secondary object image.  This prevents the subregion from having zero
%%% pixels (which cannot be measured in subsequent measure modules) in the
%%% cases where the secondary object is exactly the same size as the
%%% primary object.
ErodedPrimaryObjectImage = imerode(PrimaryObjectImage, ones(3));
SubregionObjectImage = max(0,SecondaryObjectImage - ErodedPrimaryObjectImage);

%%% Calculates object outlines
MaxFilteredImage = ordfilt2(SubregionObjectImage,9,ones(3,3),'symmetric');
%%% Determines the outlines.
IntensityOutlines = SubregionObjectImage - MaxFilteredImage;
%%% Converts to logical.
warning off MATLAB:conversionToLogical
FinalOutline = logical(IntensityOutlines);
warning on MATLAB:conversionToLogical

if ~isfield(handles.Measurements,SubregionObjectName)
    handles.Measurements.(SubregionObjectName) = {};
end

%%% This line creates two rows containing all values for both label matrix
%%% images. It then takes the unique rows (no repeats), and sorts them
%%% according to the first column which is the sub object values.
ChildParentList = sortrows(unique([SubregionObjectImage(:) SecondaryObjectImage(:)],'rows'),1);
%%% We want to get rid of the children values and keep the parent values.
ParentList = ChildParentList(:,2);
%%% This gets rid of all parent values which have no corresponding children
%%% values (where children = 0 but parent = 1).
for i = 1:max(ChildParentList(:,1))
    ParentValue = max(ParentList(ChildParentList(:,1) == i));
    if isempty(ParentValue)
        ParentValue = 0;
    end
    FinalParentList(i,1) = ParentValue;
end

if exist('FinalParentList','var')
    if max(SubregionObjectImage(:)) ~= size(FinalParentList,1)
        error(['Image processing was canceled in the ', ModuleName, ' module because a subobject cannot have two parents, something is wrong.']);
    end
    if isfield(handles.Measurements.(SubregionObjectName),'ParentFeatures')
        if handles.Current.SetBeingAnalyzed == 1
            NewColumn = length(handles.Measurements.(SubregionObjectName).ParentFeatures) + 1;
            handles.Measurements.(SubregionObjectName).ParentFeatures(NewColumn) = {SecondaryObjectName};
            handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,NewColumn) = FinalParentList;
        else
            OldColumn = strmatch(SecondaryObjectName,handles.Measurements.(SubregionObjectName).ParentFeatures);
            if length(OldColumn) ~= 1
                error(['Image processing was canceled in the ', ModuleName, ' module because you are attempting to create the same children, please remove redundant module.']);
            end
            handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,OldColumn) = FinalParentList;
        end
    else
        handles.Measurements.(SubregionObjectName).ParentFeatures = {SecondaryObjectName};
        handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed} = FinalParentList;
    end
end

%%% This line creates two rows containing all values for both label matrix
%%% images. It then takes the unique rows (no repeats), and sorts them
%%% according to the first column which is the sub object values.
ChildParentList = sortrows(unique([SubregionObjectImage(:) PrimaryObjectImage(:)],'rows'),1);
%%% We want to get rid of the children values and keep the parent values.
ParentList = ChildParentList(:,2);
%%% This gets rid of all parent values which have no corresponding children
%%% values (where children = 0 but parent = 1).
for i = 1:max(ChildParentList(:,1))
    ParentValue = max(ParentList(ChildParentList(:,1) == i));
    if isempty(ParentValue)
        ParentValue = 0;
    end
    FinalParentList(i,1) = ParentValue;
end

if exist('FinalParentList','var')
    if max(SubregionObjectImage(:)) ~= size(FinalParentList,1)
        error(['Image processing was canceled in the ', ModuleName, ' module because a subobject cannot have two parents, something is wrong.']);
    end
    if isfield(handles.Measurements.(SubregionObjectName),'ParentFeatures')
        if handles.Current.SetBeingAnalyzed == 1
            NewColumn = length(handles.Measurements.(SubregionObjectName).ParentFeatures) + 1;
            handles.Measurements.(SubregionObjectName).ParentFeatures(NewColumn) = {PrimaryObjectName};
            handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,NewColumn) = FinalParentList;
        else
            OldColumn = strmatch(PrimaryObjectName,handles.Measurements.(SubregionObjectName).ParentFeatures);
            if length(OldColumn) ~= 1
                error(['Image processing was canceled in the ', ModuleName, ' module because you are attempting to create the same children, please remove redundant module.']);
            end
            handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,OldColumn) = FinalParentList;
        end
    else
        handles.Measurements.(SubregionObjectName).ParentFeatures = {PrimaryObjectName};
        handles.Measurements.(SubregionObjectName).Parent{handles.Current.SetBeingAnalyzed} = FinalParentList;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    ColoredLabelMatrixImage = CPlabel2rgb(handles,SubregionObjectImage);
    SecondaryObjectImage = CPlabel2rgb(handles,SecondaryObjectImage);
    PrimaryObjectImage = CPlabel2rgb(handles,PrimaryObjectImage);
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    subplot(2,2,1); CPimagesc(PrimaryObjectImage); title([PrimaryObjectName, ' Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2); CPimagesc(SecondaryObjectImage); title([SecondaryObjectName, ' Image']);
    subplot(2,2,3); CPimagesc(ColoredLabelMatrixImage); title([SubregionObjectName, ' Image']);
    subplot(2,2,4); CPimagesc(FinalOutline); title([SubregionObjectName, ' Outlines']);
    CPFixAspectRatio(PrimaryObjectImage);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent modules.
fieldname = ['Segmented', SubregionObjectName];
handles.Pipeline.(fieldname) = SubregionObjectImage;

%%% Saves the ObjectCount, i.e. the number of segmented objects.
%%% See comments for the Threshold saving above
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,SubregionObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ', SubregionObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(SubregionObjectImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(SubregionObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(SubregionObjectImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(SubregionObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

if ~strcmpi(SaveOutlines,'Do not save')
    handles.Pipeline.(SaveOutlines) = FinalOutline;
end