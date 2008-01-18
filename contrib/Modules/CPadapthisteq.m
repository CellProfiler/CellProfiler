function handles = CPadapthisteq(handles)

% Help for the CPadapthisteq module:
% Category: Contributed
%
% SHORT DESCRIPTION:
% 
% *************************************************************************
% Contrast-limited adaptive histogram equalization (CLAHE)
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

functions = cell(1,6);
value = cell(1,6);


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What image do you want to apply adapthisteq filter? 
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu


%textVAR02 = What do you want to call the resulting image? 
%defaultVAR02 = adapthisteqImageName
%infotypeVAR02 = imagegroup indep
adapthisteqImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});


%textVAR03 = Parameters: 

%textVAR04

%textVAR05 = NumTiles 
%choiceVAR05 = Not Selected
%choiceVAR05 = Selected
%inputtypeVAR05 = popupmenu
NumTilesSelection = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Number of tiles by row - M (at least 2) 
%defaultVAR06 = 8
m_NumTiles = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Number of tiles by column - N (at least 2) 
%defaultVAR07 = 8
n_NumTiles = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08

%textVAR09 = ClipLimit 
%choiceVAR09 = Not Selected
%choiceVAR09 = Selected
%inputtypeVAR09 = popupmenu
ClipLimitSelection = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Contrast enhancement limit (in the range [0 1]). Higher numbers result in more contrast 
%defaultVAR10 = 0.01
n_ClipLimit = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11

%textVAR12 = NBins 
%choiceVAR12 = Not Selected
%choiceVAR12 = Selected
%inputtypeVAR12 = popupmenu
NBinsSelection = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Number of bins for the histogram. Higher values result in greater dynamic range at the cost of slower processing speed 
%defaultVAR13 = 256
n_NBins = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%textVAR14

%textVAR15 = Range 
%choiceVAR15 = Not Selected
%choiceVAR15 = Selected
%inputtypeVAR15 = popupmenu
RangeSelection = char(handles.Settings.VariableValues{CurrentModuleNum,15});

%textVAR16 =  Range of the output image data. Original: range is limited to the range of the original image. Full: full range of the output image class is used 
%choiceVAR16 = full
%choiceVAR16 = original
%inputtypeVAR16 = popupmenu
Range_param = char(handles.Settings.VariableValues{CurrentModuleNum,16});

%textVAR17

%textVAR18 = Distribution 
%choiceVAR18 = Not Selected
%choiceVAR18 = Selected
%inputtypeVAR18 = popupmenu
DistributionSelection = char(handles.Settings.VariableValues{CurrentModuleNum,18});

%textVAR19 = Desired histogram shape for the image tiles. uniform: flat histogram. Rayleigh: bell-shaped histogram . exponential: Curved histogram 
%choiceVAR19 = uniform
%choiceVAR19 = rayleigh
%choiceVAR19 = exponential
%inputtypeVAR19 = popupmenu
Distribution_param = char(handles.Settings.VariableValues{CurrentModuleNum,19});

%textVAR20 = Alfa Distribution parameter. Nonnegative real scalar 
%specifying a distribution parameter (only for rayleigh and exponential) 
%defaultVAR20 = 0.4
n_Alpha = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,20}));
%%% Reads the images.
%OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow
index=0;
if strcmp('Selected',NumTilesSelection)
    index=index+1;
    functions{1,index} = 'NumTiles';
    value{1,index}=[m_NumTiles n_NumTiles];
end
if strcmp('Selected',ClipLimitSelection)
    index=index+1;
    functions{1,index} = 'ClipLimit';
    value{1,index}=n_ClipLimit;
end
if strcmp('Selected',NBinsSelection)
    index=index+1;
    functions{1,index} = 'NBins';
    value{1,index}=n_NBins;
end
if strcmp('Selected',RangeSelection)
    index=index+1;
    functions{1,index} = 'Range';
    value{1,index}=Range_param;
end
if strcmp('Selected',DistributionSelection)
    index=index+1;
    functions{1,index} = 'Distribution';
    value{1,index}=Distribution_param;
    if ~strcmp('uniform',Distribution_param)
        index=index+1;
        functions{1,index} = 'Alpha';
        value{1,index}=n_Alpha;
    end
end


FinalImage = OrigImage;
switch(index)
    case 0
        FinalImage = adapthisteq(FinalImage);
	case 1
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1});
    case 2
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1},functions{1,2},value{1,2});
    case 3
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1},functions{1,2},value{1,2},functions{1,3},value{1,3});
    case 4
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1},functions{1,2},value{1,2},functions{1,3},value{1,3},functions{1,4},value{1,4});
    case 5
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1},functions{1,2},value{1,2},functions{1,3},value{1,3},functions{1,4},value{1,4},functions{1,5},value{1,5});
    case 6
        FinalImage = adapthisteq(FinalImage,functions{1,1},value{1,1},functions{1,2},value{1,2},functions{1,3},value{1,3},functions{1,4},value{1,4},functions{1,5},value{1,5},functions{1,6},value{1,6});
end



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
    CPimagesc(FinalImage,handles);
    title('CLAHE Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(adapthisteqImageName) = FinalImage;
