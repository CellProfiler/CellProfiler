function handles = Convolution(handles)

% Help for the Convolution module:
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

%textVAR01 = What did you call the image to be corrected? 
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image? 
%defaultVAR02 = ConvolutedImage
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What filter do you want to perform? 
%choiceVAR03 = average
%choiceVAR03 = disk
%choiceVAR03 = gaussian
%choiceVAR03 = laplacian
%choiceVAR03 = log
%choiceVAR03 = motion
%choiceVAR03 = prewitt
%choiceVAR03 = sobel
%choiceVAR03 = unsharp
%inputtypeVAR03 = popupmenu
Functions{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%% La funzione convoluzione sarà conv2(imagenostra, valore h) dove valore
%%% h è l'output della funzione fspecial(). La funzione fspecial ha come
%%% parametri le scelte della VAR03

%textVAR04 = Settings 

%textVAR05 = For AVERAGE: 

%textVAR06 = hsize rows:
%defaultVAR06 = 3
hsize_rows_avg = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = hsize columns: 
%defaultVAR07 = 3
hsize_columns_avg = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = For DISK: 

%textVAR09 = radius: 
%defaultVAR09 = 5
radius = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = For GAUSSIAN: 

%textVAR11 = hsize rows: 
%defaultVAR11 = 3
hsize_rows_gauss = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,11}));

%textVAR12 = hsize columns: 
%defaultVAR12 = 3
hsize_columns_gauss = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,12}));

%textVAR13 = sigma: 
%defaultVAR13 = 0.5
sigma_gauss = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%textVAR14 = For LAPLACIAN: 

%textVAR15 = alpha [0.0:1.0]: 
%defaultVAR15 = 0.2
alpha_lap = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,15}));

%textVAR16 = For LOG: 

%textVAR17 = hsize rows: 
%defaultVAR17 = 5
hsize_rows_log = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,17}));

%textVAR18 = hsize columns: 
%defaultVAR18 = 5
hsize_columns_log = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,18}));

%textVAR19 = sigma: 
%defaultVAR19 = 0.5
sigma_log = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,19}));

%textVAR20 = For MOTION: 

%textVAR21 = len: 
%defaultVAR21 = 9
len_motion = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,21}));

%textVAR22 = theta: 
%defaultVAR22 = 0
theta_motion = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,22}));

%textVAR23 = For UNSHARP: 

%textVAR24 = sigma: 
%defaultVAR24 = 0.2
alpha_unsharp = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,24}));

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
h=[];
if strcmp(Functions{1}, 'average')
    hsize = [hsize_rows_avg hsize_columns_avg];
    h=fspecial('average',hsize);
elseif strcmp(Functions{1}, 'disk')
    h=fspecial('disk',radius);
elseif strcmp(Functions{1}, 'gaussian')
    hsize_gauss = [hsize_rows_gauss hsize_columns_gauss]
    h=fspecial('gaussian',hsize_gauss,sigma_gauss);    
elseif strcmp(Functions{1}, 'laplacian')
    h=fspecial('laplacian',alpha_lap);  
elseif strcmp(Functions{1}, 'log')
    hsize_log = [hsize_rows_log hsize_columns_log]
    h=fspecial('log',hsize_log,sigma_log);    
elseif strcmp(Functions{1}, 'motion')
    h=fspecial('motion',len_motion,theta_motion);  
elseif strcmp(Functions{1}, 'prewitt')
    h=fspecial('prewitt');
elseif strcmp(Functions{1}, 'sobel')
    h=fspecial('sobel');
elseif strcmp(Functions{1}, 'unsharp')
    h=fspecial('unsharp',alpha_unsharp); 
end

CorrectedImage = conv2(OrigImage,h);


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
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(CorrectedImage,handles);
    title('Convoluted Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;
