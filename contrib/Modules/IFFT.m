function handles = IFFT2(handles)

% Help for the IFFT2 module:
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
% Authors:
%   Paolo Arcaini
%   Massimo Manara
%
% Website: http://www.cellprofiler.org
%
% DATA:

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What image do you want to apply IFFT2? 
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu


%textVAR02 = What do you want to call the IFFT2 image? 
%defaultVAR02 = IFFTImage
%infotypeVAR02 = imagegroup indep
ComplementedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%% Reads the images.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

FinalImage = OrigImage;
%Shift zero-frequency component of discrete Fourier transform to center of
%spectrum
FinalImage=ifft2(FinalImage);
%FinalImage=fft2(FinalImage);
%FinalImage = log(abs(FinalImage));



%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    %%% Questo andrebbe passato all'handles da parte del modulo FFT. Solo
    %%% se serve per la stampa
    OrigImage_stmp = fftshift(OrigImage);
    OrigImage_stmp = log(abs(OrigImage_stmp));
    %%%-Fine
    CPimagesc(OrigImage_stmp,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(FinalImage,handles);
    title('IFFT Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(ComplementedImageName) = FinalImage;
