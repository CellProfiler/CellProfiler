function handles = Subtract(handles)

% Help for the Subtract module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Subtracts the intensities of one image from another.
% *************************************************************************
%
% Settings:
% Subtracting may substantially change the range of pixel intensities in
% the resulting image, so each image can be multiplied by a factor prior to
% subtracting. This factor can be a positive number.
%
% Do you want negative values in the image to be set to zero?
% Values outside the range of 0 to 1 might not be handled well by other
% modules. Here, you have the option of setting negative values to 0.
% For other options (e.g. setting values over 1 to equal 1), see the
% Rescale Intensity module.
%
% See also SubtractBackground, RescaleIntensity.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Subtract this image:
%infotypeVAR01 = imagegroup
SubtractImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = From this image:
%infotypeVAR02 = imagegroup
BasicImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = SubtractedCellStain
%infotypeVAR03 = imagegroup indep
ResultingImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the factor to multiply the first image by before subtracting:
%defaultVAR04 = 1
MultiplyFactor1 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter the factor to multiply the second image by before subtracting:
%defaultVAR05 = 1
MultiplyFactor2 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Do you want negative values in the image to be set to zero?
%choiceVAR06 = Yes
%choiceVAR06 = No
Truncate = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the images you want to analyze and assigns them to
%%% variables.
BasicImage = CPretrieveimage(handles,BasicImageName,ModuleName,'MustBeGray','CheckScale');

SubtractImage = CPretrieveimage(handles,SubtractImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

ResultingImage = imsubtract(MultiplyFactor2*BasicImage,MultiplyFactor1*SubtractImage);
if strcmpi(Truncate,'Yes')
    ResultingImage(ResultingImage < 0) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(BasicImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); 
    CPimagesc(BasicImage,handles); 
    title([BasicImageName, ' image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); 
    CPimagesc(SubtractImage,handles); 
    title([SubtractImageName, ' image']);
    subplot(2,2,3); 
    CPimagesc(ResultingImage,handles); 
    title([BasicImageName,' minus ',SubtractImageName,' = ',ResultingImageName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure.
handles.Pipeline.(ResultingImageName) = ResultingImage;