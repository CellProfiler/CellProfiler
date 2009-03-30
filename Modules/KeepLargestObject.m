function handles = KeepLargestObject(handles)

% Help for KeepLargestObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
%
% If there is more than one primary object inside a secondary object,
% delete all except the largest one.
%
% *************************************************************************

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

% PyCP Notes: 
% Anne 3-26-09: I think this module should be a new option within
% FilterByObjectMeasurement. We could add this one simple case, or we could
% add the more flexible "Keep the single object with the lowest or highest
% value of which measurement?" I think this module functions only on
% primary objects inside secondary ones whereas the new functionality
% should allow things like discarding everything but the largest primary
% object in the whole image, discarding everything but the brightest
% child object inside each parent object, etc.


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
    [hImage,hAx] = CPimagesc(LargestRGB,handles,ThisModuleFigureNumber);
    title(hAx,[LargestObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',LargestObjectName];
handles = CPaddimages(handles,fieldname,Largest);

handles = CPsaveObjectCount(handles, LargestObjectName, Largest);
handles = CPsaveObjectLocations(handles, LargestObjectName, Largest);
