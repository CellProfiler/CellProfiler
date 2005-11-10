function handles = Restart(handles)

% Help for the Restart module:
% Category: File Processing
%
% Restarts an analysis run where it left off.
% To use: Start CellProfiler and insert Restart as the only module.
% Click Analyze images. When the dialog "Choose a settings or output file"
% appears, select the output file of the incomplete run. Click ok and
% analysis will continue.
%
% SAVING IMAGES: The thresholded images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.

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
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = Use this module if you have canceled processing of a pipeline
%that you would like to finish. For Restart to work, it must be the only module in the pipeline. Click "Analyze Images" and when prompted, choose the OUTPUT FILE of the pipeline you would like to complete.

%%%VariableRevisionNumber = 1

if (handles.Current.SetBeingAnalyzed ~= 1)
    return
end

callback = handles.FunctionHandles.LoadPipelineCallback;
try
    [filepath, filename, errFlg, updatedhandles] = callback(gcbo,[],guidata(gcbo));
catch
    errFlg = 1;
end

if (errFlg ~= 0)
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not initialize pipeline.']);
    return;
end

try
    importhandles = load(fullfile(filepath,filename));
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not load from file ' ...
        fullfile(filepath,filename),'.']);
end
% save figure properties
ScreenSize = get(0,'ScreenSize');
ScreenWidth = ScreenSize(3);
ScreenHeight = ScreenSize(4);
fig = handles.Current.(['FigureNumberForModule',CurrentModule]);
RestartFigColor = get(fig,'color');
if ~isempty(fig)
    try
        close(fig); % Close the Restart figure
    end
end

handles.Settings = updatedhandles.Settings;
handles.Pipeline = importhandles.handles.Pipeline;
handles.Measurements = importhandles.handles.Measurements;
handles.Current = importhandles.handles.Current;
handles.VariableBox = updatedhandles.VariableBox;
handles.VariableDescription = updatedhandles.VariableDescription;
handles.Current.StartingImageSet = handles.Current.SetBeingAnalyzed + 1;
handles.Current.CurrentModuleNumber = '01';

%%% Reassign figures handles and open figure windows
userData.Application = 'CellProfiler';
for i=1:handles.Current.NumberOfModules;
    if iscellstr(handles.Settings.ModuleNames(i)) == 1
        handles.Current.(['FigureNumberForModule' TwoDigitString(i)]) = ...
            figure('name',[char(handles.Settings.ModuleNames(i)), ' Display'],...
            'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442],...
            'color',RestartFigColor,'UserData',userData);
    end
end
guidata(gcbo,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

return;
end

%%% SUBFUNCTION %%%
function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
  error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);
return;
end

