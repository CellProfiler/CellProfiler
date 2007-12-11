function handles = KeepLargestObjects(handles)

% Help for KeepLargestObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% If there is more than one primary object inside a secondary object,
% delete all except the largest one.
% *************************************************************************
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

%textVAR01 = What did you call the primary objects?
%infotypeVAR01 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the secondary objects?
%infotypeVAR02 = objectgroup
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the largest primary objects?
%defaultVAR03 = LargestObjects
%infotypeVAR03 = objectgroup indep
LargestObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

Primary = CPretrieveimage(handles,['Segmented', PrimaryObjectName],ModuleName);
Secondary = CPretrieveimage(handles,['Segmented', SecondaryObjectName],ModuleName);


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

labels = Secondary;
labels(Primary == 0) = 0;
Largest = CPlargestComponents(labels);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    LargestRGB = CPlabel2rgb(handles,Largest);

    CPfigure(handles,'Image',ThisModuleFigureNumber);
%    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%        CPresizefigure(Orig,'TwoByOne',ThisModuleFigureNumber)
%    end
    CPimagesc(LargestRGB,handles);
    title([LargestObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',LargestObjectName];
handles.Pipeline.(fieldname) = Largest;

handles = CPsaveObjectCount(handles, LargestObjectName, Largest);
handles = CPsaveObjectLocations(handles, LargestObjectName, Largest);
