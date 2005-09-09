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
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$





%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = No data required for this module.

%%%VariableRevisionNumber = 1

drawnow
if (handles.Current.SetBeingAnalyzed ~= 1)
    return;
end;

callback = handles.LoadPipelineButton;
try
    [filepath, filename, errFlg, updatedhandles] = callback(gcbo,[],guidata(gcbo));
catch
    errFlg = 1;
end;

if (errFlg ~= 0)
    error('Processing cancelled because module Restart could not initialize pipeline.');
    return;
end;

try
    importhandles = load(fullfile(filepath,filename));
catch
    error(['Processing cancelled because module Restart could not load from file ' ...
        fullfile(filepath,filename),'.']);
end;
% save figure properties
ScreenSize = get(0,'ScreenSize');
ScreenWidth = ScreenSize(3);
ScreenHeight = ScreenSize(4);
fig = handles.Current.(['FigureNumberForModule',CurrentModule]);
RestartFigColor = get(fig,'color');
close(fig);   % Close the Restart figure

clear handles.Settings;
handles.Settings = updatedhandles.Settings;
clear handles.Pipeline;
handles.Pipeline = importhandles.handles.Pipeline;
clear handles.Measurements;
handles.Measurements = importhandles.handles.Measurements;
clear handles.Current;
handles.Current = importhandles.handles.Current;
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

