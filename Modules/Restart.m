function handles = Restart(handles)

% Help for the Restart module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Restarts image analysis which had failed or was canceled, using the
% partially completed output file.
% *************************************************************************
%
% Restarts an analysis run where it left off. Put Restart into a new
% pipeline with no other modules. Click Analyze images. When the dialog
% "Choose a settings or output file" appears, select the output file of the
% incomplete run. Click OK and the pipeline will load from the output file
% and analysis will continue where it left off during the partially
% completed run.

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Use this module if you have canceled processing of a pipeline that you would like to finish. For Restart to work, it must be the only module in the pipeline. Click "Analyze Images" and when prompted, choose the output file (OUT.mat) of the pipeline you would like to complete.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%
%%% PROCESSING %%%
%%%%%%%%%%%%%%%%%%
drawnow

%% Check
if (handles.Current.SetBeingAnalyzed ~= 1)
    error('Restart Module must be the only module in the pipeline')
end

callback = handles.FunctionHandles.LoadPipelineCallback;
try
    [filepath, filename, errFlg, updatedhandles] = callback(gcbo,[],guidata(gcbo));
catch
    errFlg = 1;
end

if (errFlg ~= 0)
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not initialize pipeline.']);
end

try
    importhandles = load(fullfile(filepath,filename));
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not load from file ' ...
        fullfile(filepath,filename),'.']);
end
% save figure properties
[ScreenWidth,ScreenHeight] = CPscreensize;

% Close the Restart figure
CPclosefigure(handles,CurrentModule)

handles.Settings = updatedhandles.Settings;
handles.Pipeline = importhandles.handles.Pipeline;
handles.Measurements = importhandles.handles.Measurements;
handles.Current = importhandles.handles.Current;
handles.VariableBox = updatedhandles.VariableBox;
handles.VariableDescription = updatedhandles.VariableDescription;
handles.Current.StartingImageSet = handles.Current.SetBeingAnalyzed + 1;
handles.Current.CurrentModuleNumber = '01';
handles.Preferences.DisplayWindows = importhandles.handles.Preferences.DisplayWindows;

%%% Reassign figures handles and open figure windows
for i=1:handles.Current.NumberOfModules;
    if iscellstr(handles.Settings.ModuleNames(i))
        if handles.Preferences.DisplayWindows(i)
            handles.Current.(['FigureNumberForModule' CPtwodigitstring(i)]) = ...
                CPfigure(handles,'','name',[char(handles.Settings.ModuleNames(i)), ' Display, cycle # '],...
                'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442]);
        end
    end
end
guidata(gcbo,handles);

drawnow
