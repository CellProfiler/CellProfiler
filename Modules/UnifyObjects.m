function handles = UnifyObjects(handles)

% Help for UnifyObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Relabels objects so that objects within a specified distance of each
% other have the same label.
% *************************************************************************
%
% If the distance threshold is zero, only objects that are touching
% will be unified.
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for licesne details and copyright
% information.  See the file AUTHORS for contributors.
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the relabeled objects?
%defaultVAR02 = UnifiedObjects
%infotypeVAR02 = objectgroup indep
RelabeledObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Distance within which objects should be unified
%defaultVAR03 = 0
DistanceThreshold = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,3})); %#ok Ignore MLint

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

Orig = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName);


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

dilated = imdilate(Orig > 0, strel('disk', DistanceThreshold));
merged = bwlabel(dilated);
merged(Orig == 0) = 0;
Relabeled = merged;


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    OrigRGB = CPlabel2rgb(handles,Orig);
    RelabeledRGB = CPlabel2rgb(handles,Relabeled);

    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(Orig,'TwoByOne',ThisModuleFigureNumber)
    end
    subplot(2,1,1);
    CPimagesc(OrigRGB,handles);
    title([ObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2);
    CPimagesc(RelabeledRGB,handles);
    title(RelabeledObjectName);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',RelabeledObjectName];
handles.Pipeline.(fieldname) = Relabeled;

if ~isfield(handles.Measurements,RelabeledObjectName)
    handles.Measurements.(RelabeledObjectName) = {};
end

handles = CPsaveObjectCount(handles, RelabeledObjectName, Relabeled);
handles = CPsaveObjectLocations(handles, RelabeledObjectName, Relabeled);

