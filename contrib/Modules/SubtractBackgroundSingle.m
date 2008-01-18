function handles = SubtractBackgroundSingle(handles)

% Help for the Subtract Background module:
% Category: Contributed
%
% SHORT DESCRIPTION:
% 
% *************************************************************************
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% DATA:

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be corrected? 
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image? 
%defaultVAR02 = SubBackSingleImage
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What n^th pixel do you want to use as threshold? 
%defaultVAR03 = 10
PixelThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

index=handles.Current.SetBeingAnalyzed;
fieldname = ['Pathname', ImageName];
try Pathname = handles.Pipeline.(fieldname);
catch error(['Image processing was canceled in the ', ModuleName, ' module because it must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Subtract Background module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from the ', ModuleName,' module onward.'])
end
%%% Retrieves the list of filenames where the images are stored from the
%%% handles structure.
fieldname = ['FileList', ImageName];
FileList = handles.Pipeline.(fieldname);
if size(FileList,1) == 2
error(['Image processing was canceled in the ', ModuleName, ' module because it cannot function on movies.']);
end
[TempImage, handles] = CPimread(fullfile(Pathname,char(FileList(index))), handles);
SortedColumnImage = sort(reshape(TempImage, [],1));
TenthMinimumPixelValue = SortedColumnImage(PixelThreshold); 
if TenthMinimumPixelValue ~= 0
    %%% Subtracts the MinimumTenthMinimumPixelValue from every pixel in the
    %%% original image.  This strategy is similar to that used for the "Apply
    %%% Threshold and Shift" module.
    CorrectedImage = OrigImage - TenthMinimumPixelValue;
    %%% Values below zero are set to zero.
    CorrectedImage(CorrectedImage < 0) = 0;

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
        end
        %%% A subplot of the figure window is set to display the original
        %%% image, some intermediate images, and the final corrected image.
        subplot(2,1,1); 
        CPimagesc(OrigImage,handles);
        title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% The mean image does not absolutely have to be present in order to
        %%% carry out the calculations if the illumination image is provided,
        %%% so the following subplot is only shown if MeanImage exists in the
        %%% workspace.
        subplot(2,1,2); 
        CPimagesc(CorrectedImage,handles);
        title('Corrected Image');
        %%% Displays the text.
        if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText'))
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [0 0 200 20],'fontname','helvetica','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
        else
            displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
        end
        displaytext = ['Background threshold used: ', num2str(TenthMinimumPixelValue)];
        set(displaytexthandle,'string',displaytext)
    end
else CorrectedImage = OrigImage;
end % This end goes with the if MinimumTenthMinimumPixelValue ~= 0 line above.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;
