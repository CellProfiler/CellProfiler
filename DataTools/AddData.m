function handles = AddData(handles)

% Help for the Add Data tool:
% Category: Data Tools
%
% Use this tool if you would like text information about each image to
% be recorded in the output file along with measurements (e.g. Gene
% names, accession numbers, or sample numbers). You will then be
% guided through the process of choosing a text file that contains the
% text data for each image. More than one set of text information can
% be entered for each image; each set of text will be a separate
% column in the output file.        
%
% See also CLEARDATA VIEWDATA.

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

    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne
CurrentDir = pwd;
    cd(handles.Current.DefaultOutputDirectory)

ExistingOrMemory = CPquestdlg('Do you want to add sample info into an existing output file or into memory so that it is incorporated into future output files?', 'Load Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    OutputFile = []; pOutName = []; fOutName = [];
else [fOutName,pOutName] = uigetfile('*.mat','Add sample info to which existing output file?');
% else [fOutName,pOutName] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'*.mat'),'Add sample info to which existing output file?');
    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne


%%% Allows canceling.
    if fOutName == 0
        return
    else
        try OutputFile = load(fullfile(pOutName fOutName));
        catch error('Sorry, the file could not be loaded for some reason.')
        end
    end
end

%%% Opens a dialog box to retrieve a file name that contains a list of
%%% sample descriptions, like gene names or sample numbers.
[fname,pname] = uigetfile({'*.csv;*.txt','Readable files: .txt (Plain text) or .csv (Comma-separated)'},'Choose sample info file');
%%% If the user presses "Cancel", the fname will = 0 and nothing will
%%% happen.
if fname == 0
else extension = fname(end-2:end);
    HeadingsPresent = CPquestdlg('Does the first row of your file contain headings?', 'Are headings present?', 'Yes', 'No', 'Cancel', 'Yes');
    %%% Allows canceling.
    if strcmp(HeadingsPresent, 'Cancel') == 1 | isempty(HeadingsPresent) == 1
        return
    end
    %%% Determines the file type.
    if strcmp(extension,'csv') == 1
        try fid = fopen([pname fname]);
            FirstLineOfFile = fgetl(fid);
            LocationOfCommas = strfind(FirstLineOfFile,',');
            NumberOfColumns = size(LocationOfCommas,2) + 1;
            Format = repmat('%s',1,NumberOfColumns);
            %%% Returns to the beginning of the file so that textscan
            %%% reads the entire contents.
            frewind(fid);
            ImportedData = textscan(fid,Format,'delimiter',',');
            for i = 1:NumberOfColumns
                ColumnOfData = ImportedData{i};
                ColumnOfData = ColumnOfData';
                %%% Sends the heading and the sample info to a
                %%% subfunction to be previewed and saved.
                if i == 1
                    Newhandles = handles;
                end
                [Newhandles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(Newhandles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
                if CancelOption == 1
                    fclose(fid);
                    warndlg('None of the sample info was saved.')
                    return
                end
            end
            fclose(fid);
            if strcmp(ExistingOrMemory,'Memory') == 1
                %%% For future output files:
                %%% Saves the new sample info to the handles
                %%% structure.
                handles = Newhandles;
                h = CPmsgbox(['The sample info is successfully stored in memory and will be added to future output files']);
                waitfor(h)
            else
                %%% For existing output files:
                %%% Saves the output file with this new sample info.
                save([pOutName,fOutName],'-struct','OutputFile');
                h = CPmsgbox(['The sample info was successfully added to output file']);
                waitfor(h)
            end
        catch lasterr
            fclose(fid)
            if CancelOption == 1
                fclose(fid);
            else error('Sorry, the sample info could not be imported for some reason.')
                fclose(fid);
            end
        end
    elseif strcmp(extension,'txt') == 1
        try fid = fopen([pname fname]);
            ImportedData = textscan(fid,'%s','delimiter','\r');
            ColumnOfData = ImportedData{1};
            ColumnOfData = ColumnOfData';
            %%% Sends the heading and the sample info to a
            %%% subfunction to be previewed and saved.
            [Newhandles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(handles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
            if CancelOption == 1
                fclose(fid);
                warndlg('None of the sample info was saved.')
                return
            end
            fclose(fid);
            if strcmp(ExistingOrMemory,'Memory') == 1
                %%% For future output files:
                %%% Saves the new sample info to the handles
                %%% structure.
                handles = Newhandles;
                h = CPmsgbox(['The sample info will be added to future output files']);
                waitfor(h)
            else
                %%% For existing output files:
                %%% Saves the output file with this new sample info.
                save([pOutName,fOutName],'-struct','OutputFile');
                h = CPmsgbox(['The sample info was successfully added to output file']);
                waitfor(h)
            end
        catch lasterr
            fclose(fid)
            if CancelOption == 1
                fclose(fid);
            else error('Sorry, the sample info could not be imported for some reason.')
                fclose(fid);
            end
        end
    else errordlg('Sorry, the list of sample descriptions must be in a text file (.txt) or comma delimited file (.csv).');
    end
end

cd(CurrentDir)
    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne


%%% SUBFUNCTION %%%
function [handles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(handles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
%%% Sets the initial value to zero.
CancelOption = 0;
%%% Extracts the sample info and the headings from the first row, if they are present.
if strcmp(HeadingsPresent, 'Yes') == 1
    SingleHeading = ColumnOfData(1);
    %%% Converts to char in order to perform the following lines.
    SingleHeading = char(SingleHeading);
    %%% Replaces spaces with underscores, because spaces
    %%% are forbidden in fieldnames.
    SingleHeading(strfind(SingleHeading,' ')) = '_';
    %%% Strips weird characters (e.g. punctuation) out, because such
    %%% characters are forbidden in fieldnames.  The user is still
    %%% responsible for making sure their heading begins with a letter
    %%% rather than a number or underscore.
    PermittedCharacterLocations = regexp(SingleHeading, '[A-Za-z0-9_]');
    SingleHeading = SingleHeading(PermittedCharacterLocations);
    if isempty(SingleHeading) == 1
        SingleHeading = 'Heading not yet entered';
    end
    %%% Converts back to cell array.
    SingleHeading = {SingleHeading};
    ColumnOfSampleInfo = ColumnOfData(2:end);
else SingleHeading = {'Heading not yet entered'};
    ColumnOfSampleInfo = ColumnOfData(1:end);
end
NumberSamples = length(ColumnOfSampleInfo);
%%% Displays a notice.  The buttons don't do anything except proceed.
Notice = {['You have ', num2str(NumberSamples), ' lines of sample information with the heading:']; ...
    char(SingleHeading); ...
    ''; ...
    'The next window will show you a preview of the sample'; ...
    'info you have loaded, and you will then have the'; ...
    'opportunity to enter or change the heading name (Disallowed';...
    'characters have been removed). Press ''OK'' to continue.';...
    '-------'; ...
    'Please note:';...
    '(1) For text files, any spaces or punctuation characters'; ...
    'may split the entry into two entries.';...
    '(2) For csv files, entries containing commas ';...
    'will split the entry into two entries.';...
    '(3) Check that the order of the image files within Matlab is';...
    'as expected.  For example, If you are running Matlab within';...
    'X Windows on a Macintosh, the Mac will show the files as: ';...
    '(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) whereas the X windows ';...
    'system that Matlab uses will show them as ';...
    '(1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9) and so on.  So be sure that ';...
    'the order of your sample info matches the order that Matlab ';...
    'is using.  You can see the order of image files within';...
    'CellProfiler, or in the Current Directory window of';...
    'Matlab. Go to View > Current Directory to open the window';...
    'if it is not already visible.'};
[Selection,OK] = listdlg('ListString',Notice,'ListSize', [300 600],'Name','Imported sample info',...
    'PromptString','Press any button to continue.',...
    'SelectionMode','single');
%%% Allows canceling.
if OK == 0
    CancelOption = 1;
else
    %%% Displays a filenameslistbox so the user can preview the data.  The buttons in
    %%% this filenameslistbox window don't do anything except proceed.
    [Selection,OK] = listdlg('ListString',ColumnOfSampleInfo, 'ListSize', [300 600],...
        'Name','Sample info preview',...
        'PromptString','Press ''OK'' to continue.',...
        'SelectionMode','single');
    %%% Allows canceling.
    if OK == 0
        CancelOption = 1;
    else
        %%% Sets the initial value.
        HeadingApproved = 0;
        while HeadingApproved ~= 1
            if strcmp(SingleHeading, 'Heading not yet entered') == 1;
                SingleHeading = {''};
            end
            %%% The input dialog displays the current candidate for
            %%% the heading, or it is blank if nothing has been
            %%% entered.
            SingleHeading = inputdlg('Enter the heading for these sample descriptions (e.g. GeneNames                 or SampleNumber). Your entry must be one word with letters and                   numbers only, and must begin with a letter.','Name the Sample Info',1,SingleHeading);
            %%% Allows canceling.
            if isempty(SingleHeading) == 1
                CancelOption = 1;
                return
            elseif strcmp(SingleHeading,'') == 1
                errordlg('No heading was entered. Please try again.');
                %%% For future output files:
            elseif strcmp(ExistingOrMemory, 'Memory') == 1
                %%% Checks to see if the heading exists already.
                if isfield(handles.Measurements, ['Imported',char(SingleHeading)]) == 1
                    Answer = CPquestdlg('Sample info with that heading already exists in memory.  Do you want to overwrite?');
                    %%% Allows canceling.
                    if isempty(Answer) == 1 | strcmp(Answer,'Cancel') == 1
                        CancelOption = 1;
                        return
                    end
                else Answer = 'Newfield';
                end
                %%% If the user does not want to overwrite, try again.
                if strcmp(Answer,'No')

                elseif strcmp(Answer,'Yes') == 1 | strcmp(Answer, 'Newfield') == 1
                    if strcmp(Answer,'Yes') == 1
                        handles.Measurements = rmfield(handles.Measurements, ['Imported',char(SingleHeading)]);
                        guidata(gcbo,handles)
                    end
                    %%% Tries to make a field with that name.
                    try handles.Measurements.(['Imported',char(SingleHeading)]) = [];
                        HeadingApproved = 1;
                    catch
                        MessageHandle = errordlg(['The heading name ',char(SingleHeading),' is not acceptable for some reason. Please try again.']);
                        waitfor(MessageHandle)
                    end
                end
            else %%% For existing output files:
                %%% Checks to see if the heading exists already. Some
                %%% old output files may not have the 'Measurements'
                %%% substructure, so we check for that field first.
                if isfield(OutputFile.handles, 'Measurements') == 1
                    if isfield(OutputFile.handles.Measurements, ['Imported',char(SingleHeading)]) == 1
                        Answer = CPquestdlg(['Sample info with the heading ',char(SingleHeading),' already exists in the output file.  Do you want to overwrite?']);
                        %%% Allows canceling.
                        if isempty(Answer) == 1 | strcmp(Answer,'Cancel') == 1
                            CancelOption = 1;
                            return
                        end
                    else Answer = 'Newfield';
                    end
                else Answer = 'Newfield';
                end
                %%% If the user does not want to overwrite, try again.
                if strcmp(Answer,'No')

                elseif strcmp(Answer,'Yes') == 1 | strcmp(Answer, 'Newfield') == 1
                    if strcmp(Answer,'Yes') == 1
                        OutputFile.handles.Measurements = rmfield(OutputFile.handles.Measurements,['Imported',char(SingleHeading)]);
                    end
                    %%% Tries to make a field with that name.
                    try OutputFile.handles.Measurements.(['Imported',char(SingleHeading)]) = [];
                        HeadingApproved = 1;
                    catch
                        MessageHandle = errordlg(['The heading name ',char(SingleHeading),' is not acceptable for some reason. Please try again.']);
                        waitfor(MessageHandle)
                    end
                end
            end
        end
        %%% Saves the sample info to the handles structure or existing output
        %%% file.
        if strcmp(ExistingOrMemory, 'Memory') == 1
            %%% For future files:
            %%% Saves the column of sample info to the handles.
            handles.Measurements.(['Imported',char(SingleHeading)]) = ColumnOfSampleInfo;
            guidata(gcbo,handles)
        else
            %%% For an existing file:
            %%% Saves the column of sample info to the handles structure from an existing output file.
            OutputFile.handles.Measurements.(['Imported',char(SingleHeading)]) = ColumnOfSampleInfo;
        end
    end
end

% Some random advice from Ganesh:
% SampleNames is a n x m (n - number of rows, m - 1 column) cell array
% If you want to make it into a 1x1 cell array where the cell element
% contains the text, you can do the following
%
% cell_data = {strvcat(SampleNames)};
%
% This will assign the string matrix being created into a single cell
% element.