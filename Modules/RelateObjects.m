function handles = RelateObjects(handles)

% Help for the Flip module:
% Category: Object Processing
%
% See also <nothing relevant>.

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
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What objects are to become subobjects?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
SubObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What are the parent objects?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ParentName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
fieldname = ['Segmented', SubObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Create Subobjects module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Create Subobjects module that the primary objects were named ', SubObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
end
SubObjectLabelMatrix = handles.Pipeline.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
fieldname = ['Segmented', ParentName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Create Subobjects module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Create Subobjects module that the primary objects were named ', ParentName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
end
ParentObjectLabelMatrix = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% This line creates two rows containing all values for both label matrix
%%% images. It then takes the unique rows (no repeats), and sorts them
%%% according to the first column which is the sub object values.
ChildParentList = sortrows(unique([SubObjectLabelMatrix(:) ParentObjectLabelMatrix(:)],'rows'),1);
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
if max(SubObjectLabelMatrix(:)) ~= size(FinalParentList,1)
    error('A subobject cannot have two parents, something is wrong.');
end

if isfield(handles.Measurements.(SubObjectName),'ParentFeatures')
    if handles.Current.SetBeingAnalyzed == 1
        NewColumn = length(handles.Measurements.(SubObjectName).ParentFeatures) + 1;
        handles.Measurements.(SubObjectName).ParentFeatures(NewColumn) = {ParentName};
        handles.Measurements.(SubObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,NewColumn) = FinalParentList;
    else
        OldColumn = strmatch(ParentName,handles.Measurements.(SubObjectName).ParentFeatures);
        if length(OldColumn) ~= 1
            error('You are attempting to create the same children, please remove redundant module.');
        end
        handles.Measurements.(SubObjectName).Parent{handles.Current.SetBeingAnalyzed}(:,OldColumn) = FinalParentList;
    end
else
    handles.Measurements.(SubObjectName).ParentFeatures = {ParentName};
    handles.Measurements.(SubObjectName).Parent{handles.Current.SetBeingAnalyzed} = FinalParentList;
end

for i = 1:max(ParentList)
    ChildList(i,1) = length(FinalParentList(FinalParentList == i));
end

if isfield(handles.Measurements.(ParentName),'ChildrenFeatures')
    if handles.Current.SetBeingAnalyzed == 1
        NewColumn = length(handles.Measurements.(ParentName).ChildrenFeatures) + 1;
        handles.Measurements.(ParentName).ChildrenFeatures(NewColumn) = {[SubObjectName,' Count']};
        handles.Measurements.(ParentName).Children{handles.Current.SetBeingAnalyzed}(:,NewColumn) = ChildList;
    else
        OldColumn = strmatch([SubObjectName,' Count'],handles.Measurements.(ParentName).ChildrenFeatures);
        if length(OldColumn) ~= 1
            error('You are attempting to create the same children, please remove redundant module.');
        end
        handles.Measurements.(ParentName).Children{handles.Current.SetBeingAnalyzed}(:,OldColumn) = ChildList;
    end
else
    handles.Measurements.(ParentName).ChildrenFeatures = {[SubObjectName,' Count']};
    handles.Measurements.(ParentName).Children{handles.Current.SetBeingAnalyzed} = ChildList;
end

%%% Since the label matrix starts at zero, we must include this value in
%%% the list to produce a label matrix image with children re-labeled to
%%% their parents values. This does not get saved and is only for display.
FinalParentListLM = [0;FinalParentList];
NewObjectParentLabelMatrix = FinalParentListLM(SubObjectLabelMatrix+1);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    subplot(2,2,1);
    ColoredParentLabelMatrixImage = CPlabel2rgb(handles,ParentObjectLabelMatrix);
    imagesc(ColoredParentLabelMatrixImage);
    title(['Parent Objects, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2);
    ColoredSubObjectLabelMatrixImage = CPlabel2rgb(handles,SubObjectLabelMatrix);
    imagesc(ColoredSubObjectLabelMatrixImage);
    title('Original Sub Objects');
    subplot(2,2,3);
    ColoredNewObjectParentLabelMatrix = CPlabel2rgb(handles,NewObjectParentLabelMatrix);
    imagesc(ColoredNewObjectParentLabelMatrix);
    title('New Sub Objects');
end