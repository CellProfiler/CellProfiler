function handles =  InverseRadonTransform(handles)

% Help for the InverseRadonTransform module:
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
%defaultVAR02 = IRadonImage
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = theta range[0,179]: 
%defaultVAR03 = 45
theta = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Settings 
%choiceVAR04 = NO
%choiceVAR04 = YES
%inputtypeVAR04 = popupmenu
Functions{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For interpolation: 

%textVAR06 = What type do you want to perform? 
%choiceVAR06 = nearest
%choiceVAR06 = linear
%choiceVAR06 = spline
%inputtypeVAR06 = popupmenu
Functions{1} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = For filter: 

%textVAR08 = What filter do you want to perform? 
%choiceVAR08 = Ram-Lak
%choiceVAR08 = Shepp-Logan
%choiceVAR08 = Cosine
%choiceVAR08 = Hamming
%choiceVAR08 = Hann
%inputtypeVAR08 = popupmenu
Functions{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For frequency scaling: 

%textVAR10 = frequency scaling range: [0,1]: 
%defaultVAR10 = 1
fs = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = For output size: 

%textVAR12 = output size: 
%defaultVAR12 = 2*floor(size(R,1)/(2*sqrt(2)))
os = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,12}));

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
A=[];
if strcmp(Functions{3}, 'NO')
    A = iradon(OrigImage,theta);
elseif strcmp(Functions{1}, 'nearest')
    elseif strcmp(Functions{2}, 'Ram-Lak')
        A = iradon(OrigImage,theta,'nearest','Ram-Lak',fs,os );
        elseif strcmp(Functions{2}, 'Shepp-Logan')
            A = iradon(OrigImage,theta,'nearest','Shepp-Logan',fs,os );
                elseif strcmp(Functions{2}, 'Cosine')
                    A = iradon(OrigImage,theta,'nearest','Cosine',fs,os );
                         elseif strcmp(Functions{2}, 'Hamming')
                            A = iradon(OrigImage,theta,'nearest','Hamming',fs,os );
                                elseif strcmp(Functions{2}, 'Hann')
                                    A = iradon(OrigImage,theta,'nearest','Hann',fs,os );
      elseif strcmp(Functions{1}, 'linear')
        elseif strcmp(Functions{2}, 'Ram-Lak')
        A = iradon(OrigImage,theta,'linear','Ram-Lak',fs,os );
        elseif strcmp(Functions{2}, 'Shepp-Logan')
            A = iradon(OrigImage,theta,'linear','Shepp-Logan',fs,os );
                elseif strcmp(Functions{2}, 'Cosine')
                    A = iradon(OrigImage,theta,'linear','Cosine',fs,os );
                         elseif strcmp(Functions{2}, 'Hamming')
                            A = iradon(OrigImage,theta,'linear','Hamming',fs,os );
                                elseif strcmp(Functions{2}, 'Hann')
                                    A = iradon(OrigImage,theta,'linear','Hann',fs,os );
          
        elseif strcmp(Functions{1}, 'spline')
         elseif strcmp(Functions{2}, 'Ram-Lak')
        A = iradon(OrigImage,theta,'spline','Ram-Lak',fs,os );
        elseif strcmp(Functions{2}, 'Shepp-Logan')
            A = iradon(OrigImage,theta,'spline','Shepp-Logan',fs,os );
                elseif strcmp(Functions{2}, 'Cosine')
                    A = iradon(OrigImage,theta,'spline','Cosine',fs,os );
                         elseif strcmp(Functions{2}, 'Hamming')
                            A = iradon(OrigImage,theta,'spline','Hamming',fs,os );
                                elseif strcmp(Functions{2}, 'Hann')
                                    A = iradon(OrigImage,theta,'spline','Hann',fs,os );
end                                    
     
CorrectedImage = A;
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
    title('IRadon Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;
