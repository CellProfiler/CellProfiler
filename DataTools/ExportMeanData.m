function handles = ExportMeanData(handles)

% Help for the Export Mean Data tool:
% Category: Data Tools
%
% Once image analysis is complete, use this tool to select the
% output file to extract the measurements and other information about
% the analysis.  The data will be converted to a delimited text file
% which can be read by most programs.  By naming the file with the
% extension for Microsoft Excel (.xls), the file is usually easily
% openable by Excel.
%
% See also EXPORTCELLBYCELL.

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

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
else
    LoadedHandles = load(fullfile(RawPathname, RawFileName));
    %%% Extract the fieldnames of measurements from the handles structure.
    Fieldnames = fieldnames(LoadedHandles.handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5) == 1);
    FileFieldNames = Fieldnames(strncmp(Fieldnames, 'Filename', 8) == 1);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    TimeElapsedFieldNames = Fieldnames(strncmp(Fieldnames, 'TimeElapsed', 11) == 1);
    
    %%% Error detection.
    if isempty(MeasFieldnames) && isempty(FileFieldNames) && isempty(ImportedFieldnames) && isempty(TimeElapsedFieldNames)
        errordlg('No measurements were found in the file you selected. In the handles structure contained within the output file, the Measurements substructure must have fieldnames prefixed by ''Image'', ''Imported'', ''Filename'', or ''TimeElapsed''.')
    else
        %%% Tries to determine the number of image sets for which there are data.
        if isempty(FileFieldNames{1}) == 0
        fieldname = FileFieldNames{1};
        elseif isempty(MeasFieldnames{1}) == 0
        fieldname = MeasFieldnames{1};
        elseif isempty(ImportedFieldnames{1}) == 0
        fieldname = ImportedFieldnames{1};
        elseif isempty(TimeElapsedFieldNames{1}) == 0
        fieldname = TimeElapsedFieldNames{1};
        end
        TotalNumberImageSets = num2str(length(LoadedHandles.handles.Measurements.(fieldname)));
        TotalNumberImageSetsMsg = ['As a shortcut,                     type the numeral 0 to extract data from all ', TotalNumberImageSets, ' image sets.'];
        %%% Ask the user to specify the number of image sets to extract.
        NumberOfImages = inputdlg({['How many image sets do you want to extract?  ',TotalNumberImageSetsMsg]},'Specify number of image sets',1,{'0';' '});
        %%% If the user presses the Cancel button, the program goes to the end.
        if isempty(NumberOfImages)
        else
            %%% Calculate the appropriate number of image sets.
            NumberOfImages = str2double(NumberOfImages{1});
            if NumberOfImages == 0
                NumberOfImages = str2double(TotalNumberImageSets);
            elseif NumberOfImages > length(LoadedHandles.handles.Measurements.(char(MeasFieldnames(1))));
                errordlg(['There are only ', length(LoadedHandles.handles.Measurements.(char(MeasFieldnames(1)))), ' image sets total.'])
                %%% TODO: This error checking is only for the first field of
                %%% measurements.  Should make it more comprehensive.
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
            FileName = inputdlg('What do you want to call the resulting measurements file?  It will be saved in the default output directory, as specified in the main CellProfiler window. To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
            %%% If the user presses the Cancel button, the program goes to the end.
            if isempty(FileName)
            else
                FileName = FileName{1};
                OutputFileOverwrite = exist([cd,'/',FileName],'file'); %%% TODO: Fix filename construction.
                if OutputFileOverwrite ~= 0
                    errordlg('A file with that name already exists in the default output directory.  Repeat and choose a different file name.')
                else
                    %%% Extract the measurements.  Waitbar shows the percentage of image sets
                    %%% remaining.
                    WaitbarHandle = waitbar(0,'Extracting measurements...');
                    %%% TODO: Combine all the fieldnames into a single
                    %%% variable to speed this processing.
                    
                    %%% Preallocate the variable Measurements.
                    NumberOfMeasFieldnames = length(MeasFieldnames);
                    NumberOfFileFieldNames = length(FileFieldNames);
                    NumberOfImportedFieldnames = length(ImportedFieldnames);
                    NumberOfTimeElapsedFieldNames = length(TimeElapsedFieldNames);
                    
                    NumberOfFields = NumberOfMeasFieldnames + NumberOfFileFieldNames + NumberOfImportedFieldnames + NumberOfTimeElapsedFieldNames;
                    Measurements(NumberOfImages,NumberOfFields) = {[]};
                    %%% Finished preallocating the variable Measurements.
                    TimeStart = clock;
                    for imagenumber = 1:NumberOfImages
                        FieldNumber = 0;
                        for FileNameFieldNumber = 1:NumberOfFileFieldNames
                            Fieldname = cell2mat(FileFieldNames(FileNameFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber,FieldNumber) = {LoadedHandles.handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for ImportedFieldNumber = 1:NumberOfImportedFieldnames
                            Fieldname = cell2mat(ImportedFieldnames(ImportedFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber,FieldNumber) = {LoadedHandles.handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for MeasFieldNumber = 1:NumberOfMeasFieldnames
                            Fieldname = cell2mat(MeasFieldnames(MeasFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber,FieldNumber) = {LoadedHandles.handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for TimeElapsedFieldNumber = 1:NumberOfTimeElapsedFieldNames
                            Fieldname = cell2mat(TimeElapsedFieldNames(TimeElapsedFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            %%% Merged batch files don't include timing information
                            if (length(LoadedHandles.handles.Measurements.(Fieldname)) >= imagenumber),
                                Measurements(imagenumber, FieldNumber) = {LoadedHandles.handles.Measurements.(Fieldname){imagenumber}};
                            end
                        end
                        CurrentTime = clock;
                        TimeSoFar = etime(CurrentTime,TimeStart);
                        TimePerSet = TimeSoFar/imagenumber;
                        ImagesRemaining = NumberOfImages - imagenumber;
                        TimeRemaining = round(TimePerSet*ImagesRemaining);
                        WaitbarText = ['Extracting measurements... ', num2str(TimeRemaining), ' seconds remaining.'];
                        waitbar(imagenumber/NumberOfImages, WaitbarHandle, WaitbarText)
                    end
                    close(WaitbarHandle) 
                    %%% Open the file and name it appropriately.
                    fid = fopen(FileName, 'wt');
                    %%% Write the MeasFieldnames as headings for columns.
                    for i = 1:NumberOfFileFieldNames
                        fwrite(fid, char(FileFieldNames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfImportedFieldnames
                        fwrite(fid, char(ImportedFieldnames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfMeasFieldnames
                        fwrite(fid, char(MeasFieldnames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfTimeElapsedFieldNames
                        fwrite(fid, char(TimeElapsedFieldNames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end

                    fwrite(fid, sprintf('\n'), 'char');
                    %%% Write the Measurements.
                    WaitbarHandle = waitbar(0,'Writing the measurements file...');
                    NumberMeasurements = size(Measurements,1);
                    TimeStart = clock;
                    for i = 1:NumberMeasurements
                        for measure = 1:NumberOfFields
                            val = Measurements(i,measure);
                            val = val{1};
                            if ischar(val),
                                fwrite(fid, sprintf('%s\t', val), 'char');
                            elseif isempty(val)
                                fwrite(fid, sprintf('%s\t', ''), 'char');
                            else
                                fwrite(fid, sprintf('%g\t', val), 'char');
                            end
                        end
                        fwrite(fid, sprintf('\n'), 'char');
                        CurrentTime = clock;
                        TimeSoFar = etime(CurrentTime,TimeStart);
                        TimePerSet = TimeSoFar/i;
                        ImagesRemaining = NumberOfImages - i;
                        TimeRemaining = round(TimePerSet*ImagesRemaining);
                        WaitbarText = ['Writing the measurements file... ', num2str(TimeRemaining), ' seconds remaining.'];
                        waitbar(i/NumberMeasurements, WaitbarHandle, WaitbarText)
                    end
                    close(WaitbarHandle) 
                    %%% Close the file
                    fclose(fid);
                    helpdlg(['The file ', FileName, ' has been written to the default output directory.'])
                end % This goes with the error catching at the beginning of the file.
            end % This goes with the error catching "No measurements found" at the beginning of the file.
        end % This goes with the "Cancel" button on the Number of Image Sets dialog.
    end % This goes with the "Cancel" button on the FileName dialog.
end % This goes with the "Cancel" button on the RawFileName dialog.

cd(handles.Current.StartupDirectory);
% In case I want to save data that is 
% all numbers, with different numbers of rows for each column, the
% following code might be helpful:
% fid = fopen(filename,'w');
% for i = 1:length(Measurements)
%     fprintf(fid,'%d   ',Measurements{i});
%     fprintf(fid,'\n');
% end
% fclose(fid);
% type eval(filename)