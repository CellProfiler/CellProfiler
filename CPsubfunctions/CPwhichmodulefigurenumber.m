function ThisModuleFigureNumber = CPwhichmodulefigurenumber(CurrentModule)
%%% Looks up the proper figure number based on the Current Module number.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);