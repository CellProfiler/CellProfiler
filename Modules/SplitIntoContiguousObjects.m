function handles = SplitIntoContiguousObjects(handles)

% Help for SplitIntoContiguousObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% 
% *************************************************************************
%
% A new "measurement" will be added for each input object.  This
% "measurement" is a number that indicates the relabeled object
% number.
%

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision:$

%%% VARIABLES
drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the relabeled objects?
%defaultVAR02 = SplitObjects
%infotypeVAR02 = objectgroup indep
RelabeledObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

% Repeat for each of the three images by calling a subfunction that
% does the actual work.
handles = doItForObjectName(handles, 'Segmented', ObjectName, RelabeledObjectName);
if isfield(handles.Pipeline, ['SmallRemovedSegmented', ObjectName])
  handles = doItForObjectName(handles, 'SmallRemovedSegmented', ObjectName, RelabeledObjectName);
end

function handles = doItForObjectName(handles, prefix, ObjectName, RelabeledObjectName, DistanceThreshold, GrayscaleImageName)
drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

Orig = CPretrieveimage(handles, [prefix, ObjectName], ModuleName);

%%%
%%% IMAGE ANALYSIS
%%%
drawnow

Relabeled = bwlabel(Orig > 0);

% Compute the mapping from the new labels to the old labels.  This
% mapping is stored as a measurement later so we know which objects
% have been split.
Mapping = zeros(max(Relabeled(:)), 1);
props = regionprops(Relabeled, {'PixelIdxList'});
for i=1:max(Relabeled(:))
  Mapping(i,1) = Orig(props(i).PixelIdxList(1));
end

%%%
%%% DISPLAY RESULTS
%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    OrigRGB = CPlabel2rgb(handles, Orig);
    RelabeledRGB = CPlabel2rgb(handles, Relabeled);

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

%%%
%%% SAVE DATA TO HANDLES STRUCTURE
%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = [prefix, RelabeledObjectName];
handles.Pipeline.(fieldname) = Relabeled;

if ~isfield(handles.Measurements,RelabeledObjectName)
    handles.Measurements.(RelabeledObjectName) = {};
end

handles = CPsaveObjectCount(handles, RelabeledObjectName, Relabeled);
handles = CPsaveObjectLocations(handles, RelabeledObjectName, Relabeled);

handles.Measurements.(RelabeledObjectName).ContiguousObjectLabelsFeatures = { 'ObjectID' };
handles.Measurements.(RelabeledObjectName).ContiguousObjectLabels(handles.Current.SetBeingAnalyzed) = { Mapping };

