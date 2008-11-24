function handles = LoadText(handles)

% Help for the Load Text module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Loads text information corresponding to images. This data (e.g. gene
% names or sample numbers) can be displayed on a grid or exported with the
% measurements to help track samples.
% *************************************************************************
%
% Use this tool to load in text information. This is useful for two
% reasons:
% 1. Some modules, like DisplayGridInfo, place text information onto
% images. In this case, the number of text entries that you load with this
% module must be identical to the number of grid locations.
% 2. If the number of text entries that you load with this module is
% identical to the number of cycles you are processing, the text
% information you load will be placed in the output files alongside the
% measurements that are made. Therefore, the information will be exported
% with the measurements when you use the ExportData data tool, helping you
% to keep track of your samples. If you forget this module, you can also
% run the AddData data tool after processing is complete; its function is
% the same for this purpose.
%
% The text information to be loaded must be in a separate text file with
% the following syntax:
%
% DESCRIPTION <description>
% <Text info 1>
% <Text info 2>
% <Text info 3>
%              .
%              .
%
% <description> is a description of the text information stored in the
% file. It can contain spaces or unusual characters.
%
% For example:
%
% DESCRIPTION Gene names
% Gene X
% Gene Y
% Gene Z
% 
% Be sure that the file is saved in plain text format (.txt), not Rich Text
% Format (.rtf).
%
% Path Name: 
% Type period (.) for the default image folder, or ampersand (&) for the 
% default output folder.
% NOTE: this nomenclature is opposite that in SaveImages for historical purposes.
%
% See also DisplayGridInfo module and AddData data tool.

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

%filenametextVAR01 = What is the file containing the text that you want to load?
TextFileName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What would you like to call the loaded text?
%defaultVAR02 = names
%infotypeVAR02 = datagroup indep
FieldName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%pathnametextVAR03 = Enter the path name to the folder where the text file to be loaded is located.  Type period (.) for the default image folder, or ampersand (&) for the default output folder.
PathName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 2
%%% note - if this module changes its variable ordering or meaning, CalculateStatisticsDataTool needs updating.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmp(PathName,'.',1)
    if length(PathName) == 1
        PathName = handles.Current.DefaultImageDirectory;
    else
        PathName = fullfile(handles.Current.DefaultImageDirectory,strrep(strrep(PathName(2:end),'/',filesep),'\',filesep),'');
    end
elseif strncmp(PathName, '&', 1)
    if length(PathName) == 1
        PathName = handles.Current.DefaultOutputDirectory;
    else
        PathName = fullfile(handles.Current.DefaultOutputDirectory,strrep(strrep(PathName(2:end),'/',filesep),'\',filesep),'');
    end
else
    % Strip ending slash if inserted
    if strcmp(PathName(end),'/') || strcmp(PathName(end),'\'), PathName = PathName(1:end-1); end
end

%%% Note that all the text data is loaded and placed in the handles
%%% structure during the first image cycle. So if the user cancels
%%% processing earlier, there will be too many image measurements present.
%%% This is probably OK with the Export modules/data tools.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% Parse text file %%%
    fid = fopen(fullfile(PathName,TextFileName),'r');
    if fid == -1
        error(['Image processing was canceled in the ', ModuleName, ' module because the file could not be opened.  It might not exist or you might not have given its valid location. You specified this: ',fullfile(PathName,TextFileName)]);
    end

    % Get description
    s = fgets(fid,11);
    if ~strcmp(s,'DESCRIPTION')
        error(['Image processing was canceled in the ', ModuleName, ' module because the first line in the text information file is ', s, '. The first line of the file must start with DESCRIPTION.'])
    end
    Description = fgetl(fid);
    Description = Description(2:end);       % Remove space

    % Read following lines into a cell array
    Text = [];
    while 1
        s = fgetl(fid);
        if ~ischar(s), break, end               % order reversed, checks the string before it tries to replace things
        s = strrep(s,sprintf('\t'),' ');
        if ~isempty(s)
            Text{end+1} = s;
        end
    end
    fclose(fid);
    
    %%% Add the data
    for LineNumber = 1:length(Text)
        handles = CPaddmeasurements(handles,'Image',CPjoinstrings('LoadedText',FieldName),Text{LineNumber});
        %%% We need to increment the set being analyzed so the text data can be
        %%% stored for each cycle.
        handles.Current.SetBeingAnalyzed = handles.Current.SetBeingAnalyzed + 1;
    end
    %%% Set it back to 1, since this really is the first image cycle being
    %%% processed.
    handles.Current.SetBeingAnalyzed = 1; 
    
    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        drawnow
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Text',ThisModuleFigureNumber);
        uicontrol('style','text','units','normalized','HorizontalAlignment','left','string',['Description: ',Description],'position',[.05 .55 .8 .4],'BackgroundColor',[.7 .7 .9])
        uicontrol('style','text','units','normalized','HorizontalAlignment','left','string',['Text: ',Text],'position',[.05 .1 1 .7],'BackgroundColor',[.7 .7 .9])
    end
end