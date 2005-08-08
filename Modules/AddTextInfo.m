function handles = AddTextInfo(handles)

% Help for the Add Text Information:
% Category: Other
%
% Use this tool if you would like text information.  The information could
% be referring to each image set, each object, or anything else the user
% would like.
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
% <identifier> is used as field name when storing the text information in
% the Matlab structure. It must be one word. <description> is a description
% of the text information stored in the file. It can be a sentence.
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
%
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

%textVAR01 = What is the textfile that you want to add?
%choiceVAR01 = Browse while running
TextFileName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = What would you like to call the data?
%infotypeVAR02 = datagroup indep
%defaultVAR02 = names
FieldName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if handles.Current.SetBeingAnalyzed == 1
    %%% Select file with text information to be added
    if strcmp(TextFileName,'Browse while running')
        if exist(handles.Current.DefaultOutputDirectory, 'dir')
            [filename, pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'*.*'),'Pick file with text information');
        else
            [filename, pathname] = uigetfile('*.*','Pick file with text information');
        end

        if filename == 0 %User canceled
            return;
        end
    else
        [pathname filename extension]=fileparts(TextFileName);
        filename = [filename extension];
    end

    %%% Parse text file %%%
    fid = fopen(fullfile(pathname,filename),'r');

    if fid == -1
        errordlg('Could not open file.  It might not exist or you might not have given its valid path.');
    end


    % Get description
    s = fgets(fid,11);
    if ~strcmp(s,'DESCRIPTION')
        errordlg('The second line in the text information file must start with DESCRIPTION')
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
    if ~isfield(handles.Measurements.Image,FieldName)
        handles.Measurements.Image.([FieldName,'Text']) = {Description};
        handles.Measurements.Image.(FieldName) = Text;
     %%% If the entered field already exists we have to append to this field
    else
        handles.Measurements.Image.([FieldName,'Text']) = cat(2,handles.Measurements.Image.([FieldName,'Text']),{Description});
        handles.Measurements.Image.(FieldName) = cat(2,handles.Measurements.Image.(FieldName),Text);
    end
end



