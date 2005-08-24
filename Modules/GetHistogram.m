function handles = GetHistogram(handles)
% Help for the Get Histogram module:
% Category: Other

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to include?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the generated histograms?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = OrigHist
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = How many bins do you want?
%choiceVAR03 = Automatic
%choiceVAR03 = 2
%choiceVAR03 = 16
%choiceVAR03 = 256
NumBins = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Set the range for frequency counts
%choiceVAR04 = Automatic
FreqRange = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Log transform the histogram?
%choiceVAR05 = No
%choiceVAR05 = Yes
LogOption = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Enter any optional commands or leave a period.
OptionalCmds = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%defaultVAR06 = .

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING:
% figure, imshow(BlurredImage, []), title('BlurredImage')
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING:
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

OrigImage= handles.Pipeline.(ImageName);

if strcmp(LogOption,'Yes')
    OrigImage(OrigImage == 0) = min(OrigImage(OrigImage > 0));
    OrigImage = log(OrigImage);
    hi = max(OrigImage(:));
    lo = min(OrigImage(:));
    OrigImage = (OrigImage - lo)/(hi - lo);
end

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
drawnow

HistHandle = CPfigure(handles,ThisModuleFigureNumber);
if strcmp(NumBins,'Automatic')
    imhist(OrigImage);
else
    imhist(OrigImage, str2double(NumBins));
end

if ~strcmp(FreqRange,'Automatic')
    YRange = strread(FreqRange);
    ylim(YRange);
end

if ~strcmp(OptionalCmds, '.')
    eval(OptionalCmds);
end

OneFrame = getframe(HistHandle);
handles.Pipeline.(HistImage)=OneFrame.cdata;
close (ThisModuleFigureNumber);
