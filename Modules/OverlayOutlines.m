function handles = OverlayOutlines(handles)

% Help for the Overlay Outlines module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Places outlines produced by an identify module over a desired image.
% *************************************************************************
%
% Outlines (in a special format produced by an identify module) can be
% placed on any desired image (grayscale or color) and then this resulting
% image can be saved using the SaveImages module.
%
% Settings:
% Would you like to set the intensity (brightness) of the outlines to be
% the same as the brightest point in the image, or the maximum possible
% value for this image format?
% If your image is quite dim, then putting bright white lines onto it may
% not be useful. It may be preferable to make the outlines equal to the
% maximal brightness already occurring in the image.
%
% See also identify modules.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = On which image would you like to display the outlines?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the outlines that you would like to display?
%infotypeVAR02 = outlinegroup
OutlineName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = Would you like to set the intensity (brightness) of the outlines to be the same as the brightest point in the image, or the maximum possible value for this image format?
%choiceVAR03 = Max of image
%choiceVAR03 = Max possible
MaxType = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the image with the outlines displayed?
%defaultVAR04 = Do not save
%infotypeVAR04 = imagegroup indep
SavedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For color images, what do you want the color of the outlines to be?
%choiceVAR05 = White
%choiceVAR05 = Black
%choiceVAR05 = Red
%choiceVAR05 = Green
%choiceVAR05 = Blue
%choiceVAR05 = Yellow
%inputtypeVAR05 = popupmenu
OutlineColor = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage = CPretrieveimage(handles,ImageName,ModuleName);
OutlineImage = CPretrieveimage(handles,OutlineName,ModuleName,'MustBeGray','DontCheckScale',size(OrigImage));

if size(OrigImage,3) ~= 3
    if strcmp(MaxType,'Max of image')
        ValueToUseForOutlines = max(max(OrigImage));
    elseif strcmp(MaxType,'Max possible')
        if isfloat(OrigImage(1,1))
            ValueToUseForOutlines=1;
        else
            ValueToUseForOutlines = intmax(class(OrigImage(1,1)));
        end
    else
        error(['Image processing was canceled in the ', ModuleName, ' module because the value of MaxType was not recognized.']);
    end

    NewImage = OrigImage;
    NewImage(OutlineImage ~= 0) = ValueToUseForOutlines;
else
    Color1 = 1;
    Color2 = 1;
    Color3 = 1;
    if strcmpi(OutlineColor,'Black')
        Color1 = 0;
        Color2 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Red')
        Color2 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Green')
        Color1 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Blue')
        Color1 = 0;
        Color2 = 0;
    elseif strcmpi(OutlineColor,'Yellow')
        Color3 = 0;
    end
    NewImage1 = OrigImage(:,:,1);
    NewImage2 = OrigImage(:,:,2);
    NewImage3 = OrigImage(:,:,3);
    NewImage1(OutlineImage ~= 0) = Color1;
    NewImage2(OutlineImage ~= 0) = Color2;
    NewImage3(OutlineImage ~= 0) = Color3;
    NewImage(:,:,1) = NewImage1;
    NewImage(:,:,2) = NewImage2;
    NewImage(:,:,3) = NewImage3;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'OneByOne',ThisModuleFigureNumber)
    end
    CPimagesc(NewImage,handles);
    title(['Original Image with Outline Overlay, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    Callback = 'string=get(gcbo,''string'');UserData=get(gcbo,''UserData''); if strcmp(string,''off''),imagesc(UserData{1});set(gcbo,''string'',''on'');elseif strcmp(string,''on''),imagesc(UserData{2});set(gcbo,''string'',''off'');else,set(gcbo,''string'',''on'');end;clear UserData string;';
    uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
        'UserData',{OrigImage NewImage},'backgroundcolor',[.7 .7 .9],...
        'Callback',Callback);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if ~strcmp(SavedImageName,'Do not save')
    handles.Pipeline.(SavedImageName) = NewImage;
end