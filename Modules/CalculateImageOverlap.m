function handles = CalculateImageOverlap(handles)

% Help for the Calculate Image Overlap module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module takes two binary images, one defined as ground truth and
% one the result of an algorithm, and finds the true positive, true
% negative, false positive, and false negative areas.  The F-factor is
% calculated from these areas.
% *************************************************************************
% Settings:
% "Which image represents the ground truth?" : This image is a binary (ie
% masked) image in which user-identified objects are represented.
% "Which image do you want to test against the ground truth?" : This image
% is a binary (ie masked) image which is the result of some image
% processing algorithm (either in CellProfiler or any image processing
% software) that you would like to compare with the ground truth image.  
%
% The module calculates the overlap of the two image sets, and determines
% the F-factor, a measure of the algorithm's precision and recall.
%
% Note: If you want to use the output of this module in a subsequesnt
% calculation, we suggest you specify the output name rather than use
% Automatic naming.
%
% 

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

%textVAR01 = Which image represents the ground truth (true positives)?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
GroundTruthName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which image do you want to test against the ground truth?
%infotypeVAR02 = imagegroup
%inputtypeVAR02 = popupmenu
TestImgName = char(handles.Settings.VariableValues{CurrentModuleNum,2});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

GroundTruth = CPretrieveimage(handles,GroundTruthName,ModuleName, 'DontCheckColor', 'DontCheckScale');

TestImg = CPretrieveimage(handles,TestImgName,ModuleName, 'DontCheckColor', 'DontCheckScale');

%%Threshold Images
TruePosImg = GroundTruth>0;

TestPosImg = TestImg>0;

TrueNegImg = GroundTruth<1;

%% Subtract images to find the overlap
%add a try catch in case the images are different sizes  -KateTODO
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

FalsePosRate = FalsePosPixels./(FalsePosPixels + TrueNegPixels);

FalseNegRate = FalseNegPixels./(PixelsBelongingtoClass);

Precision = TruePosPixels./PixelsLabeledinClass;

Recall = TruePosPixels./PixelsBelongingtoClass;

Ffactor = (2.*(Precision.*Recall))./(Precision+Recall);

%% Save
handles = CPaddmeasurements(handles,'Image','Ffactor',Ffactor);
handles = CPaddmeasurements(handles,'Image','Precision',Precision);
handles = CPaddmeasurements(handles,'Image','Recall',Recall);
handles = CPaddmeasurements(handles,'Image','FalsePosRate',FalsePosRate);
handles = CPaddmeasurements(handles,'Image','FalseNegRate',FalseNegRate);


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
       
            ax{1} = subplot(2,3,1,'Parent',ThisModuleFigureNumber);
            CPimagesc(TruePosImg,handles,ax{1});
            title(['True Positive Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)],'Parent',ax{1});
       
            ax{2} = subplot(2,3,2,'Parent',ThisModuleFigureNumber);
            CPimagesc(TrueNegImg,handles,ax{2});
            title(['True Negative Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)],'Parent',ax{2});
         
            ax{3} = subplot(2,3,3,'Parent',ThisModuleFigureNumber);
            CPimagesc(FalsePosImg,handles,ax{3});
            title(['False Positive Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)],'Parent',ax{3});
            
            ax{4} = subplot(2,3,4,'Parent',ThisModuleFigureNumber);
            CPimagesc(FalseNegImg,handles,ax{4});
            title(['False Negative Image, cycle #',num2str(handles.Current.SetBeingAnalyzed)],'Parent',ax{4});
           
            % Place the text in position 5
            posx = get(ax{2},'Position');
            posy = get(ax{4},'Position');
            pos = [posx(1)+0.05 posy(2)+posy(4) posx(3)+0.1 0.04];
                
                bgcolor = get(ThisModuleFigureNumber,'Color');
                str{1} =    ['Precision: ' num2str(Precision,'%4.2f')];
                str{2} =    ['Recall: ' num2str(Recall, '%4.2f')];
                str{3} =    ['F-factor: ',num2str(Ffactor,'%4.2f')];
                str{4} =    ['False Positive Rate: ',num2str(FalsePosRate,'%4.2f')];
                str{5} =    ['False Negative Rate:',num2str(FalseNegRate,'%4.2f')];
                

                for i = 1:length(str),
                    h = uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[pos(1) pos(2)-0.04*i pos(3:4)],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',str{i},'FontSize',handles.Preferences.FontSize);
                end
end
end

           
        
   
           
        
   