function handles = OverlayOutlines(handles)

% Help for the Overlay Outlines module:
% Category: Other
%
% There is no help right now.
%
% See also n/a.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision: 1718 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%


%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = On what image would you like to display the outlines?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What are the outlines that you would like to display?
%infotypeVAR02 = outlinegroup
OutlineName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = Would you like to set the outlines to be the maximum in the image, or the maximum possible value?
%choiceVAR03 = Max of Image
%choiceVAR03 = Max Possible
MaxType = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu


%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
    OrigImage = handles.Pipeline.(ImageName);
catch
    error(['Cannot load ' ImageName ' from pipeline.  Make sure you run a load image module, before this one.']);
end

try
    OutlineImage = handles.Pipeline.(OutlineName);
catch
    error(['Cannot load ' OutlineName ' from pipeline.  Make sure you saved the outlines in an earlier module.']);
end

if any(size(OrigImage) ~= size(OutlineImage))
    error(['The sizes of the image, ' size(OrigImage) ' is not the same as the size of the outlines, ' size(OutlineImage)]);
end

if strcmp(MaxType,'Max of Image')
    m = max(max(OrigImage));
elseif strcmp(MaxType,'Max Possible')
    if isfloat(OrigImage(1,1))
        m=1;
    else
        m = intmax(class(OrigImage(1,1)));
    end
else
    error('The value of MaxType was not recognized');
end
    
NewImage = OrigImage;
NewImage(OutlineImage ~= 0) = m;




%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
    
FigHandle = CPfigure(ThisModuleFigureNumber);

imagesc(NewImage);

uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
    'UserData',{OrigImage NewImage},'backgroundcolor',[.7 .7 .9],...
    'Callback','string=get(gcbo,''string'');UserData=get(gcbo,''UserData''); if strcmp(string,''off''),imagesc(UserData{1});set(gcbo,''string'',''on'');elseif strcmp(string,''on''),imagesc(UserData{2});set(gcbo,''string'',''off'');else,set(gcbo,''string'',''on'');end;clear UserData string;');







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

