function handles = CreateWebPage(handles)

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%infotypeVAR01 = imagegroup
%textVAR01 = What did you call the full-size images you want to include?
OrigImage = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%infotypeVAR02 = imagegroup
%textVAR02 = What did you call the thumbnail images you want to include?
ThumbImage = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the HTML file with the images above?
FileName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%defaultVAR03 = images1.html

%textVAR04 = HTML file save directory
%choiceVAR04 = One level over the images
%choiceVAR04 = Same as the images
DirectoryOption = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Webpage title
PageTitle = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%defaultVAR05 = CellProfiler Images

%textVAR06 = Choose the background color, or provide the html color code (e.g. #00FF00)
%choiceVAR06 = White
%choiceVAR06 = Black
%choiceVAR06 = Aqua
%choiceVAR06 = Blue
%choiceVAR06 = Fuchsia
%choiceVAR06 = Green
%choiceVAR06 = Gray
%choiceVAR06 = Lime
%choiceVAR06 = Maroon
%choiceVAR06 = Navy
%choiceVAR06 = Olive
%choiceVAR06 = Purple
%choiceVAR06 = Red
%choiceVAR06 = Silver
%choiceVAR06 = Teal
%choiceVAR06 = Yellow
BGColor = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Number of columns for thumbnails
%choiceVAR07 = 1
%choiceVAR07 = 2
%choiceVAR07 = 3
%choiceVAR07 = 4
%choiceVAR07 = 5
ThumbCols = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));
%inputtypeVAR07 = popupmenu custom

%textVAR08 = Table border width
%choiceVAR08 = 0
%choiceVAR08 = 1
%choiceVAR08 = 2
TableBorderWidth = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = Choose the table border color, or provide the html color code (e.g. #00FF00)
%choiceVAR09 = White
%choiceVAR09 = Black
%choiceVAR09 = Aqua
%choiceVAR09 = Blue
%choiceVAR09 = Fuchsia
%choiceVAR09 = Green
%choiceVAR09 = Gray
%choiceVAR09 = Lime
%choiceVAR09 = Maroon
%choiceVAR09 = Navy
%choiceVAR09 = Olive
%choiceVAR09 = Purple
%choiceVAR09 = Red
%choiceVAR09 = Silver
%choiceVAR09 = Teal
%choiceVAR09 = Yellow
TableBorderColor = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu custom

%textVAR10 = Spacing between thumbnails
%choiceVAR10 = 0
%choiceVAR10 = 1
%choiceVAR10 = 2
ThumbSpacing = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu custom

%textVAR11 = Thumbnail border width
%choiceVAR11 = 0
%choiceVAR11 = 1
%choiceVAR11 = 2
ThumbBorderWidth = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu custom

%textVAR12 = Create new window for each click on thumbnail
%choiceVAR12 = Once only
%choiceVAR12 = For each image
%choiceVAR12 = No
CreateNewWindow = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING: 
% figure, imshow(BlurredImage, []), title('BlurredImage') 
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING: 
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

if SetBeingAnalyzed == 1

    NumOrigImage = numel(handles.Pipeline.(['FileList' OrigImage]));
    NumThumbImage = numel(handles.Pipeline.(['FileList' ThumbImage]));
    if NumOrigImage ~= NumThumbImage
        msgbox('Number of original images and thumbnail images do not match');
        return;
    end
    OrigImageFileNames = handles.Pipeline.(['FileList' OrigImage]);
    OrigImagePathName = handles.Pipeline.(['Pathname' OrigImage]);
    ThumbImageFileNames = handles.Pipeline.(['FileList' ThumbImage]);
    ThumbImagePathName = handles.Pipeline.(['Pathname' ThumbImage]);
    CurrentImage = 1;
    
    if strcmp(DirectoryOption,'One level over the images')
        LastDirPos = max(findstr('\',OrigImagePathName))+1;
        if isempty(LastDirPos)
            LastDirPos = max(findstr('/',OrigImagePathName))+1;
        end
        
        HTMLSavePath = OrigImagePathName(1:LastDirPos-2);
        OrigImagePathName = OrigImagePathName(LastDirPos:end);
        ThumbImagePathName = ThumbImagePathName(LastDirPos:end);
    else
        HTMLSavePath = OrigImagePathName;
        OrigImagePathName = '';
        ThumbImagePathName = '';
    end
    
  %  OrigMinPath = OrigImagePathNames{1};
  %  ThumbMinPath = ThumbImagePathNames{1};
  %  for i=1:NumOrigImage
   %     if length(OrigImagePathNames{i})<length(OrigMinPath)
  %          OrigMinPath = OrigImagePathNames{i};
   %     end
   %     if length(ThumbImagePathNames{i})<length(ThumbMinPath)
    %        ThumbMinPath = ThumbImagePathNames{i};
   %     end        
   % end
   % 
   % for i=1:NumOrigImage
   %     OrigImagePathNames{i}=OrigImagePathNames{i}(length(OrigMinPath)+1:end);
    %    ThumbImagePathNames{i}=ThumbImagePathNames{i}(length(ThumbMinPath)+1:end);
   % end
    
    WindowName = '_CPNewWindow';
    
    Lines = '<HTML>';
    Lines = strvcat(Lines,['<HEAD><TITLE>',PageTitle,'</TITLE></HEAD>']);
    Lines = strvcat(Lines,['<BODY BGCOLOR=',AddQ(BGColor),'>']);
    Lines = strvcat(Lines,['<TABLE BORDER=',TableBorderWidth, ' BORDERCOLOR=', AddQ(TableBorderColor), ' CELLPADDING=0',' CELLSPACING=',ThumbSpacing,'>']);
    while CurrentImage <= NumOrigImage
        Lines = strvcat(Lines,'<TR>');
        for i=1:ThumbCols

            Lines = strvcat(Lines,'<TD>');
            if strcmp(CreateNewWindow,'Once only')
                Lines = strvcat(Lines,['<A HREF=',AddQ(fullfile(OrigImagePathName,OrigImageFileNames{CurrentImage})),' TARGET=',AddQ(WindowName),'>']);
            elseif strcmp(CreateNewWindow,'For each image')
                Lines = strvcat(Lines,['<A HREF=',AddQ(fullfile(OrigImagePathName,OrigImageFileNames{CurrentImage})),' TARGET=',AddQ([WindowName,num2str(CurrentImage)]),'>']);
            else
                Lines = strvcat(Lines,['<A HREF=',AddQ(fullfile(OrigImagePathName,OrigImageFileNames{CurrentImage})),'>']);
            end
            Lines = strvcat(Lines,['<IMG SRC=',AddQ(fullfile(ThumbImagePathName,ThumbImageFileNames{CurrentImage})),' BORDER=',ThumbBorderWidth,'>']);
            Lines = strvcat(Lines,'</A>');
            Lines = strvcat(Lines,'</TD>');
            CurrentImage = CurrentImage + 1;
            if CurrentImage > NumThumbImage
                break;
            end
        end
        Lines = strvcat(Lines,'</TR>');
    end
    Lines = strvcat(Lines,'</TABLE>');
    Lines = strvcat(Lines,'</BODY>');
    Lines = strvcat(Lines,'</HTML>');
    HTMLFullfile = fullfile(HTMLSavePath,FileName)
    dlmwrite(HTMLFullfile,Lines,'delimiter','');
    msgbox(['Your webpage has been saved as ', HTMLFullfile, '.']);
end

function AfterQuotation = AddQ(BeforeQuotation)
AfterQuotation = ['"',BeforeQuotation,'"'];