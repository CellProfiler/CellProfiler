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

% Variable settings for PyCP
% For exporting to database, right now ExportToDatabase assumes that
% anything loaded by load text is actually a number (so its a float) as
% opposed to a character.  Perhaps in LoadText, we could ask the user what
% they are loading- characters or numbers- and store this info (in the name
% of the loaded text? LoadedText_Char vs LoadedText_Float?) so that
% downstream modules who care can access it.  I'm guessing right now Calc.
% Stats. assumes you're giving it numbers (doses) but in theory you could
% group over any treatment and come up with stats for those, I think.
%
% Getting even more abstract about metadata, Could it be possible to load
% in your metadata w/ load text modules and have ExportToDatabase create a
% metadata table? (On the most basic level, it could just assume that
% whatever you loaded in loadtext is metadata, and created a table linked
% by image number, just with the loaded text in a separate table)
%
% I can't think of any other uses of LoadText, other than metadata, but we
% should think about what users may do with LoadText and keep it flexible
% for that too.
%
% Anne 4-9-09: I think perhaps this module should be LoadData instead of
% LoadText, since the latter implies that numbers can't be loaded. I agree
% with what Kate says above; overall this module should allow you to
% import a text file but also a tab delimited and comma delimited file. I'm
% not sure how best to handle the number vs. character issue - the options
% are asking the user, figuring it out ourselves, or requiring that the
% user put a label at the top of each column indicating numbers or
% characters.  See also the AddData data tool and Calculate Statistics
% (which relies on the input from this module).

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

	s = textscan(fid,'%s','Delimiter','\n'); s = s{:};
	%%% Get description
	if ~regexp(s{1},'^DESCRIPTION')
        error(['Image processing was canceled in the ', ModuleName, ' module because the first line in the text information file is ', s, '. The first line of the file must start with DESCRIPTION.'])
	end
	Description = regexp(s{1},'^DESCRIPTION(\s*)(?<Description>.*)','tokens','once');
	Description = Description{2};
	s = s(2:end);	% Skip to next line
	
	Text = cellfun(@strrep,s,repmat({sprintf('\t')},size(s)),repmat({' '},size(s)),'UniformOutput',false);
	fclose(fid);
	
    %%% Add the data
	handles = CPaddmeasurements(handles,'Image',CPjoinstrings('LoadedText',FieldName),Text,1:length(Text));
    
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
        uicontrol('style','text','units','normalized','HorizontalAlignment','left','string','Text:','position',[.05 .1 0.1 .7],'BackgroundColor',[.7 .7 .9])
		uicontrol('style','text','units','normalized','HorizontalAlignment','left','string',Text,	'position',[.15 .1 1 .7],'BackgroundColor',[.7 .7 .9])
    end
end