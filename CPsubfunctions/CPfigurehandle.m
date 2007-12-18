function FigureHandle = CPfigurehandle(handles)
%%% This subfunction tells you the next figure window available, based on
%%% the figures that are already allocated (but may or may not be open at
%%% the current time).

NumberOfModules = handles.Current.NumberOfModules;
for ModuleNumber = 1:NumberOfModules
    FieldName = sprintf('FigureNumberForModule%02d', ModuleNumber);
    %%% We have a 'try' here, because if we have just started a pipeline
    %%% and haven't gotten to the end yet, some figure windows may not yet
    %%% be assigned.
    try ListOfFigureNumbers(ModuleNumber) = handles.Current.(FieldName);
    end
end
NextAvailableFigureHandle = CPfigure;
close(NextAvailableFigureHandle);
FigureHandle = max([ListOfFigureNumbers + 1, NextAvailableFigureHandle]);