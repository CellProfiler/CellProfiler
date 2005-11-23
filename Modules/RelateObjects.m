function handles = RelateObjects(handles)

% Help for the RelateObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Assigns relationships: All objects (e.g. speckles) within parent object
% (e.g. nucleus) become its children. Also produces a child count for each
% parent object.
% *************************************************************************
%
% Sorry, help does not exist yet.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

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
if ~isfield(handles.Pipeline, fieldname)
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running this module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified that the primary objects were named ', SubObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The module cannot locate this image.']);
end
SubObjectLabelMatrix = handles.Pipeline.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
fieldname = ['Segmented', ParentName];
%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline, fieldname)
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running this module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified that the primary objects were named ', ParentName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The module cannot locate this image.']);
end
ParentObjectLabelMatrix = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

[handles,ChildList,FinalParentList] = CPrelateobjects(handles,SubObjectName,ParentName,SubObjectLabelMatrix,ParentObjectLabelMatrix);

%%% Since the label matrix starts at zero, we must include this value in
%%% the list to produce a label matrix image with children re-labeled to
%%% their parents values. This does not get saved and is only for display.
if ~isempty(FinalParentList)
    FinalParentListLM = [0;FinalParentList];
    NewObjectParentLabelMatrix = FinalParentListLM(SubObjectLabelMatrix+1);
else
    NewObjectParentLabelMatrix = SubObjectLabelMatrix;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    subplot(2,2,1);
    ColoredParentLabelMatrixImage = CPlabel2rgb(handles,ParentObjectLabelMatrix);
    CPimagesc(ColoredParentLabelMatrixImage);
    title(['Parent Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2);
    ColoredSubObjectLabelMatrixImage = CPlabel2rgb(handles,SubObjectLabelMatrix);
    CPimagesc(ColoredSubObjectLabelMatrixImage);
    title('Original Sub Objects');
    subplot(2,2,3);
    ColoredNewObjectParentLabelMatrix = CPlabel2rgb(handles,NewObjectParentLabelMatrix);
    CPimagesc(ColoredNewObjectParentLabelMatrix);
    title('New Sub Objects');
end