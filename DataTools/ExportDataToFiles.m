function ExportDataToFiles (handles)

ButtonSelected =questdlg('What format do you want to export to?', 'Select Export Type', 'Excel', 'SQL', 'Cancel', 'Excel')
if strcmp(ButtonSelected, 'Excel'),
    ExportExcel(handles);
elseif strcmp(ButtonSelected, 'SQL'),
    ExportSQL(handles);
else
    return;
end 

    