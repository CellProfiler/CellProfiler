function handles = ExportCellByCell(handles)

% Help for the Export Cell by Cell tool:
% Category: Data Tools
%
% This tool allows individual cell measurements to be extracted for
% a particular image and for a particular measurement parameter. The
% data will be converted to a delimited text file which can be read by
% most programs.  By naming the file with the extension for Microsoft
% Excel (.xls), the file is usually easily openable by Excel.
%
% See also EXPORTMEANDATA.

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

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
    PathToSave = handles.Current.DefaultOutputDirectory;
else
    [RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
    PathToSave = RawPathname;
end

if RawFileName == 0
    return
end
LoadedHandles = load(fullfile(RawPathname, RawFileName));

Answer = CPquestdlg('Do you want to export cell by cell data for all measurements from one image, or data from all images for one measurement?','','All measurements','All images','All measurements');

if strcmp(Answer, 'All images') == 1
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(LoadedHandles.handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        return
    end
    %%% Removes the 'Object' prefix from each name for display purposes.
    for Number = 1:length(MeasFieldnames)
        EditedMeasFieldnames{Number} = MeasFieldnames{Number}(7:end);
    end
    
    %%% Allows the user to select a measurement from the list.
    [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a measurement to export','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        return
    end
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['Object', EditedMeasurementToExtract];
    TotalNumberImageSets = length(LoadedHandles.handles.Measurements.(MeasurementToExtract));
    Measurements = LoadedHandles.handles.Measurements.(MeasurementToExtract);
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(LoadedHandles.handles.Measurements);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8) == 1 | strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'TimeElapsed',11) == 1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline or handles.Measurements structure preceded by ''Filename'', ''Imported'', or ''TimeElapsed''.')
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label each column of data with','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));

    %%% Have the user choose which of image/cells should be rows/columns
    RowColAnswer = CPquestdlg('Which layout do you want images and cells to follow in the exported data?  WARNING: Excel spreadsheets can only have 256 columns.','','Rows = Cells, Columns = Images','Rows = Images, Columns = Cells','Rows = Cells, Columns = Images');
    %%% Extracts the headings.
    try ListOfHeadings = LoadedHandles.handles.Pipeline.(HeadingToDisplay);
    catch ListOfHeadings = LoadedHandles.handles.Measurements.(HeadingToDisplay);
    end
    %%% Determines the suggested file name.
    try
        %%% Find and remove the file format extension within the original file
        %%% name, but only if it is at the end. Strip the original file format extension 
        %%% off of the file name, if it is present, otherwise, leave the original
        %%% name intact.
        CharFileName = char(RawFileName);
        PotentialDot = CharFileName(end-3:end-3);
        if strcmp(PotentialDot,'.') == 1
            BareFileName = [CharFileName(1:end-4),'.xls'];
        else BareFileName = [CharFileName,'.xls'];
        end
    catch BareFileName = 'TempData';
    end
    %%% Ask the user to name the file.
    FileName = inputdlg('What do you want to call the resulting measurements file?  To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
    %%% If the user presses the Cancel button, the program goes to the end.
    if isempty(FileName)
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist(fullfile(PathToSave,FileName),'file');
    if OutputFileOverwrite ~= 0
        Answer = CPquestdlg('A file with that name already exists in the directory ', PathToSave, '. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            return    
        end
    end
    
    %%% Opens the file and names it appropriately.
    fid = fopen(fullfile(PathToSave,FileName), 'wt');
    %%% Writes MeasurementToExtract as the heading for the first column/row.
    fwrite(fid, char(MeasurementToExtract), 'char');
    fwrite(fid, sprintf('\n'), 'char');
    
    TooWideForXLS = 0;
    if (strcmp(RowColAnswer, 'Rows = Images, Columns = Cells')),
      %%% Writes the data, row by row: one row for each image.
      for ImageNumber = 1:TotalNumberImageSets
        %%% Writes the heading in the first column.
        fwrite(fid, char(ListOfHeadings(ImageNumber)), 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
        %%% Writes the measurements for that image in successive columns.
        if (length(Measurements{ImageNumber}) > 256),
          TooWideForXLS = 1;
        end
        fprintf(fid,'%d\t',Measurements{ImageNumber});
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
      end
    else
      %%% Writes the data, row by row: one column for each image.

      % Check for truncation
      if (TotalNumberImageSets > 255),
        TooWideForXLS = 1;
      end
      
      %%% Writes the heading in the first row.
      for ImageNumber = 1:TotalNumberImageSets
        fwrite(fid, char(ListOfHeadings(ImageNumber)), 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
      end
      %%% Returns to the next row.
      fwrite(fid, sprintf('\n'), 'char');

      %%% find the number of cells in the largest set
      maxlength = 0;
      for ImageNumber = 1:TotalNumberImageSets
        maxlength = max(maxlength, length(Measurements{ImageNumber}));
      end
      
      for CellNumber = 1:maxlength,
        for ImageNumber = 1:TotalNumberImageSets
          if (length(Measurements{ImageNumber}) >= CellNumber),
            fprintf(fid, '%d\t', Measurements{ImageNumber}(CellNumber));
          else
            fprintf(fid, '\t');
          end
        end
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
      end
      %%% Closes the file
    end
    fclose(fid);
    
    if TooWideForXLS,
      helpdlg(['The file ', FileName, ' has been created in the directory ', PathToSave, '.  WARNING: This file contains more than 256 columns, and will not be readable in Excel.'])
    else 
      helpdlg(['The file ', FileName, ' has been created in the directory ', PathToSave, '.'])
    end

    
elseif strcmp(Answer, 'All measurements') == 1
    TotalNumberImageSets = LoadedHandles.handles.Current.SetBeingAnalyzed;
    %%% Asks the user to specify which image set to export.
    Answers = inputdlg({['Enter the sample number to export. There are ', num2str(TotalNumberImageSets), ' total.']},'Choose samples to export',1,{'1'});
    if isempty(Answers{1})
        return
    end
    try ImageNumber = str2double(Answers{1});
    catch errordlg('The text entered was not a number.')
        return
    end
    if ImageNumber > TotalNumberImageSets
        errordlg(['There are only ', num2str(TotalNumberImageSets), ' image sets total.'])
        return
    end
    
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(LoadedHandles.handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        return
    end
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(LoadedHandles.handles.Measurements);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1 | strncmp(Fieldnames,'Imported',8) == 1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline structure preceded by ''Filename''.')
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label this data.','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));
    %%% Extracts the headings.
    try ImageNamesToDisplay = LoadedHandles.handles.Pipeline.(HeadingToDisplay);
    catch ImageNamesToDisplay = LoadedHandles.handles.Measurements.(HeadingToDisplay);
    end
        ImageNameToDisplay = ImageNamesToDisplay(ImageNumber);
    
    %%% Determines the suggested file name.
    try
        %%% Find and remove the file format extension within the original file
        %%% name, but only if it is at the end. Strip the original file format extension 
        %%% off of the file name, if it is present, otherwise, leave the original
        %%% name intact.
        CharFileName = char(RawFileName);
        PotentialDot = CharFileName(end-3:end-3);
        if strcmp(PotentialDot,'.') == 1
            BareFileName = [CharFileName(1:end-4),'.xls'];
        else BareFileName = [CharFileName,'.xls'];
        end
    catch BareFileName = 'TempData';
    end
    %%% Asks the user to name the file.
    FileName = inputdlg('What do you want to call the resulting measurements file?  To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
    %%% If the user presses the Cancel button, the program goes to the end.
    if isempty(FileName)
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist(fullfile(PathToSave,FileName),'file');
    if OutputFileOverwrite ~= 0
        Answer = CPquestdlg('A file with that name already exists in the directory ', PathToSave,'. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            return    
        end
    end
    
    %%% Opens the file and names it appropriately.
    fid = fopen(fullfile(PathToSave,FileName), 'wt');
    %%% Writes ImageNameToDisplay as the heading for the first column/row.
    fwrite(fid, char(ImageNameToDisplay), 'char');
    fwrite(fid, sprintf('\n'), 'char');
    %%% Writes the data, row by row: one row for each measurement type.

    %%% Writes the headings
    for MeasNumber = 1:length(MeasFieldnames)
      FieldName = char(MeasFieldnames(MeasNumber));
      %%% Writes the measurement heading in the first column.
      fwrite(fid, FieldName, 'char');
      %%% Tabs to the next column.
      fwrite(fid, sprintf('\t'), 'char');
    end

    %%% Find the largest measurement set
    maxlength = 0;
    for MeasNumber = 1:length(MeasFieldnames)
      FieldName = char(MeasFieldnames(MeasNumber));
      Measurements = LoadedHandles.handles.Measurements.(FieldName);
      maxlength = max(maxlength, length(Measurements{ImageNumber}));
    end

    %%% Returns to the next row.
    fwrite(fid, sprintf('\n'), 'char');
      
    %%% Writes the data row-by-row
    for idx = 1:maxlength,
      for MeasNumber = 1:length(MeasFieldnames)
        FieldName = char(MeasFieldnames(MeasNumber));
        Measurements = LoadedHandles.handles.Measurements.(FieldName){ImageNumber};
        if (length(Measurements) >= idx),
          %%% Writes the measurements for that measurement type in successive columns.
          fprintf(fid,'%d\t',Measurements(idx));
        else
          fwrite(fid, sprintf('\t'), 'char');
        end
      end
      %%% Returns to the next row.
      fwrite(fid, sprintf('\n'), 'char');
    end

    %%% Closes the file
    fclose(fid);
    helpdlg(['The file ', FileName, ' has been created in the directory ', PathToSave])
end