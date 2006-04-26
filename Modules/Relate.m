function handles = Relate(handles)

% Help for the Relate module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Assigns relationships: All objects (e.g. speckles) within a parent object
% (e.g. nucleus) become its children.
% *************************************************************************
%
%  Allows counting the number of objects within each parent object. An
%  object will be considered a child even if the edge is the only part
%  touching a parent object. If an object is touching two parent objects,
%  the objects parent will be the higher numbered parent.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What objects are the children objects (subobjects)?
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
SubObjectLabelMatrix = CPretrieveimage(handles,['Segmented', SubObjectName],ModuleName,'MustBeGray','DontCheckScale');

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
ParentObjectLabelMatrix = CPretrieveimage(handles,['Segmented', ParentName],ModuleName,'MustBeGray','DontCheckScale');

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
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(ParentObjectLabelMatrix,'TwoByTwo',ThisModuleFigureNumber);
    end
    subplot(2,2,1);
    ColoredParentLabelMatrixImage = CPlabel2rgb(handles,ParentObjectLabelMatrix);
    CPimagesc(ColoredParentLabelMatrixImage,handles);
    title(['Parent Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2);
    ColoredSubObjectLabelMatrixImage = CPlabel2rgb(handles,SubObjectLabelMatrix);
    CPimagesc(ColoredSubObjectLabelMatrixImage,handles);
    title('Original Sub Objects');
    subplot(2,2,3);
    ColoredNewObjectParentLabelMatrix = CPlabel2rgb(handles,NewObjectParentLabelMatrix);
    CPimagesc(ColoredNewObjectParentLabelMatrix,handles);
    title('New Sub Objects');
end