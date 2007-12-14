function FigureHandle = CPfigurehandle(handles)
%%% This subfunction tells you the next figure window available, based on
%%% the figures that are already allocated (but may or may not be open at
%%% the current time).

NumberOfModules = handles.Current.NumberOfModules;
for ModuleNumber = 1:NumberOfModules
    FieldName = sprintf('FigureNumberForModule%02d', ModuleNumber);
    ListOfFigureNumbers(ModuleNumber) = handles.Current.(FieldName);
end
FigureHandle = max(ListOfFigureNumbers) + 1;