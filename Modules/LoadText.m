function handles = LoadText(handles)

% Help for the Load Text module:
% Category: File Processing
%
% Use this tool to load in text information. This is useful for certain
% modules that place text information onto images. It is also useful to
% place text information in the output files alongside the measurements
% that are made so that the information can be exported with the
% measurements.
%
% The text information must be specified in a separate text file
% with the following syntax:
%
% DESCRIPTION <description>
% <Text info>
% <Text info>
% <Text info>
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
%
% See also ADDDATA.

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

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == 1
    %%% Parse text file %%%
    fid = fopen(TextFileName,'r');
    if fid == -1
        fid = fopen(fullfile(handles.Current.DefaultImageDirectory,TextFileName),'r');
        if fid == -1
            fid = fopen(fullfile(handles.Current.DefaultOutputDirectory,TextFileName),'r');
            if fid == -1
                error(['Image processing was canceled in the ', ModuleName, ' module because the file could not be opened.  It might not exist or you might not have given its valid locaation.']);
            end
        end
    end

    % Get description
    s = fgets(fid,11);
    if ~strcmp(s,'DESCRIPTION')
        error(['Image processing was canceled in the ', ModuleName, ' module because the first line in the text information file must start with DESCRIPTION.'])
    end
    Description = fgetl(fid);
    Description = Description(2:end);       % Remove space

    % Read following lines into a cell array
    Text = [];
    while 1
        s = fgetl(fid);
        if ~ischar(s), break, end
        if ~isempty(s)
            Text{end+1} = s;
        end
    end
    fclose(fid);

    %%% Add the data
    %%% If the entered field doesn't exist  (This is the convenient way of doing it. Takes time for large ouput files??)
    if ~isfield(handles.Measurements,FieldName)
        handles.Measurements.([FieldName,'Text']) = {Description};
        handles.Measurements.(FieldName) = Text;
        %%% If the entered field already exists we have to append to this field
    else
        handles.Measurements.([FieldName,'Text']) = cat(2,handles.Measurements.([FieldName,'Text']),{Description});
        handles.Measurements.(FieldName) = cat(2,handles.Measurements.(FieldName),Text);
    end

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber) == 1
        CPfigure(handles,ThisModuleFigureNumber);
        uicontrol('style','text','units','normalized','HorizontalAlignment','left','string',['Description: ',Description],'position',[.05 .55 .8 .4],'BackgroundColor',[.7 .7 .9])
        uicontrol('style','text','units','normalized','HorizontalAlignment','left','string',['Text: ',Text],'position',[.05 .1 1 .7],'BackgroundColor',[.7 .7 .9])
    end
end