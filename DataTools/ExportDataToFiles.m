function ExportDataToFiles (handles)

ButtonSelected =questdlg('What format do you want to export to?', 'Select Export Type', 'Excel', 'SQL', 'Cancel', 'Excel')
if strcmp(ButtonSelected, 'Excel'),
    CPExportExcel(handles);
elseif strcmp(ButtonSelected, 'SQL'),
    CPExportSQL(handles);
else
    return;
end 

    