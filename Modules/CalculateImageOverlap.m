function handles = CalculateImageOverlap(handles)

% Help for the Calculate Math module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module can takes two binary images, one defined as ground truth and
% one the result of an algorithm, and finds the true positive, true
% negative, false positive, and false negative areas.  The F-factor is
% calculated from these areas.
% *************************************************************************
%
% Note: If you want to use the output of this module in a subsequesnt
% calculation, we suggest you specify the output name rather than use
% Automatic naming.
%
% See also 

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
% $Revision: 5724 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which image represents the ground truth (true positives)?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
GroundTruthName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which image do you want to test against the ground truth?
%infotypeVAR02 = imagegroup
%inputtypeVAR02 = popupmenu
TestImgName = char(handles.Settings.VariableValues{CurrentModuleNum,2});


%textVAR03 = What do you want to call the output calculated by this module? The prefix 'Overlap' will be used if you do not specify a name.
%defaultVAR03 = Automatic
OutputFeatureName = char(handles.Settings.VariableValues(CurrentModuleNum,3));
%%%VariableRevisionNumber = 6
%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

GroundTruth = CPretrieveimage(handles,GroundTruthName,ModuleName, 'MustBeBinary', 'DontCheckScale');

TestImg = CPretrieveimage(handles,TestImgName,ModuleName, 'MustBeBinary', 'DontCheckScale');

%%Threshold Images
TruePosImg = GroundTruth>0;

TestPosImg = TestImg>0;

TrueNegImg = GroundTruth<1;

%% Subtract images to find the overlap
FalsePosImg = imsubtract(TestImg,GroundTruth);

FalseNegImg = imsubtract(GroundTruth,TestImg);

%set negative pixels equal to zero
FalsePosImg(FalsePosImg<0)=0;

FalseNegImg(FalseNegImg<0)=0;

%sum up pixels to get the area
TruePosPixels = sum(TruePosImg(:));

TrueNegPixels = sum(TrueNegImg(:));

FalsePosPixels = sum(FalsePosImg(:));

FalseNegPixels = sum(FalseNegImg(:));

%% Find the Ffactor

PixelsLabeledinClass = TruePosPixels + FalsePosPixels;

PixelsBelongingtoClass = TruePosPixels + FalseNegPixels;

Precision = TruePosPixels./PixelsLabeledinClass;

Recall = TruePosPixels./PixelsBelongingtoClass;

Ffactor = (2.*(Precision.*Recall))./(Precision+Recall);

%% fill out field name
if isempty(OutputFeatureName) || strcmp(OutputFeatureName,'Automatic') == 1
    FullFeatureName = CPjoinstrings('Overlay',GroundTruthName, TestImgName);

    %% Since Matlab's max name length is 63, we need to truncate the fieldname
    TruncFeatureName = CPtruncatefeaturename(FullFeatureName);
else
    TruncFeatureName = CPjoinstrings('Overlay',OutputFeatureName);
end

%% Save
handles = CPaddmeasurements(handles,'Image',TruncFeatureName,Ffactor);


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    drawnow
    % Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);

    % Subplots of the various images to show overlap
        ax = cell(1,6);
       
            ax{1} = subplot(2,3,1);
            CPimagesc(TruePosImg,handles);
            title(['True Positive Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)]);
       
            ax{2} = subplot(2,3,2);
            CPimagesc(TrueNegImg,handles);
            title(['True Negative Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)]);
         
            ax{3} = subplot(2,3,3);
            CPimagesc(FalsePosImg,handles);
            title(['False Positive Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)]);
            
            ax{4} = subplot(2,3,4);
            CPimagesc(FalseNegImg,handles);
            title(['False Negative Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)]);
           
            % Place the text in position 5
            posx = get(ax{2},'Position');
            posy = get(ax{4},'Position');
            pos = [posx(1)+0.05 posy(2)+posy(4) posx(3)+0.1 0.04];
                
                bgcolor = get(ThisModuleFigureNumber,'Color');
                str{1} =    ['Precision: ' num2str(Precision,'%4.2f')];
                str{2} =    ['Recall: ' num2str(Recall, '%4.2f')];
                str{3} =    ['F-factor: ',num2str(Ffactor,'%4.2f')];

                for i = 1:length(str),
                    h = uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[pos(1) pos(2)-0.04*i pos(3:4)],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',str{i},'FontSize',handles.Preferences.FontSize);
                end
end

           
        
   