function handles = AlgSaturationBlurCheck1(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR1 = What did you call the image you want to check for saturation and blur?
%defaultVAR1 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
NameImageToCheck1 = handles.(fieldname);
%textVAR2 = What did you call the image you want to check for saturation and blur?
%defaultVAR2 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
NameImageToCheck2 = handles.(fieldname);
%textVAR3 = What did you call the image you want to check for saturation and blur?
%defaultVAR3 = OrigRed
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
NameImageToCheck3 = handles.(fieldname);
%textVAR4 = What did you call the image you want to check for saturation and blur?
%defaultVAR4 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
NameImageToCheck4 = handles.(fieldname);
%textVAR5 = What did you call the image you want to check for saturation and blur?
%defaultVAR5 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
NameImageToCheck5 = handles.(fieldname);
%textVAR6 = What did you call the image you want to check for saturation and blur?
%defaultVAR6 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
NameImageToCheck6 = handles.(fieldname);
%textVAR8 =  For unused colors, leave "N" in the boxes above.
%textVAR9 = Enter the blur radius, or leave 'N' to skip blur checking.
%defaultVAR9 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
BlurRadius = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
for ImageNumber = 1:6;
    ImageNumber = num2str(ImageNumber);
    %%% Read (open) the images you want to analyze and assign them to
    %%% variables.
    if strcmp(upper(eval(['NameImageToCheck',ImageNumber])), 'N') ~= 1
        fieldname = ['dOT', eval(['NameImageToCheck',ImageNumber])];
        %%% Check whether the image to be analyzed exists in the handles structure.
        if isfield(handles, fieldname) == 0
            %%% If the image is not there, an error message is produced.  The error
            %%% is not displayed: The error function halts the current function and
            %%% returns control to the calling function (the analyze all images
            %%% button callback.)  That callback recognizes that an error was
            %%% produced because of its try/catch loop and breaks out of the image
            %%% analysis loop without attempting further modules.
            error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', eval(['NameImageToCheck',ImageNumber]), ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        %%% Read the image.
        eval(['ImageToCheck',ImageNumber,' = handles.(fieldname);'])
        % figure, imshow(ImageToCheck1), title('ImageToCheck1')
        MaximumPixelValue = max(max(eval(['ImageToCheck',ImageNumber])));
        if MaximumPixelValue == 1
            eval(['Saturation',ImageNumber,' = 1;'])
        else eval(['Saturation',ImageNumber,' = 0;'])
        end
        
        %%% Check the focus of the images, if desired.
        if strcmp(upper(BlurRadius), 'N') ~= 1
            eval(['RightImage = ImageToCheck', ImageNumber, '(:,2:end);'])
            eval(['LeftImage = ImageToCheck', ImageNumber, '(:,1:end-1);'])
            eval(['FocusScore', ImageNumber, ' = std(RightImage(:) - LeftImage(:)) / mean(ImageToCheck', ImageNumber, '(:));'])
            
            %             eval(['ImageToCheck',ImageNumber,' = histeq(eval([''ImageToCheck'',ImageNumber]));'])
            %             if str2num(BlurRadius) == 0
            %                 BlurredImage = eval(['ImageToCheck',ImageNumber]);
            %             else
            %                 %%% Blurs the image.
            %                 %%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
            %                 FiltSize = max(3,ceil(4*BlurRadius));
            %                 BlurredImage = filter2(fspecial('gaussian',FiltSize, str2num(BlurRadius)), eval(['ImageToCheck',ImageNumber]));
            %                 % figure, imshow(BlurredImage, []), title('BlurredImage')
            %                 % imwrite(BlurredImage, [BareFileName,'BI','.',FileFormat], FileFormat);
            %             end
            %             %%% Subtracts the BlurredImage from the original.
            %             SubtractedImage = imsubtract(eval(['ImageToCheck',ImageNumber]), BlurredImage);
            %             handles.FocusScore(handles.setbeinganalyzed) = std(SubtractedImage(:));
            %             handles.FocusScore2(handles.setbeinganalyzed) = sum(sum(SubtractedImage.*SubtractedImage));
            %             FocusScore = handles.FocusScore
            %             FocusScore2 = handles.FocusScore2
            
        %%% Save the Focus Score  to the handles structure.  The field is named 
        %%% appropriately based on the user's input, with the 'dMT' prefix added so
        %%% that this field will be deleted at the end of the analysis batch.
        fieldname = ['dMTFocusScore', eval(['NameImageToCheck',ImageNumber])];
        handles.(fieldname)(handles.setbeinganalyzed) = {eval(['FocusScore',ImageNumber])};
        %%% Removed for parallel: guidata(gcbo, handles);
        else eval(['FocusScore', ImageNumber, ' = ''not checked'';'])
        end
        %%% Save the Saturation recording to the handles structure.  The field is named 
        %%% appropriately based on the user's input, with the 'dMT' prefix added so
        %%% that this field will be deleted at the end of the analysis batch.
        fieldname = ['dMTSaturation', eval(['NameImageToCheck',ImageNumber])];
        handles.(fieldname)(handles.setbeinganalyzed) = {eval(['Saturation',ImageNumber])};
        %%% Removed for parallel: guidata(gcbo, handles);
    else         eval(['Saturation',ImageNumber,' = ''not checked'';'])
    end
end
%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
    figure(ThisAlgFigureNumber);
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    newsize(1) = 0;
    newsize(2) = 0;
    if handles.setbeinganalyzed == 1
        newsize(3) = originalsize(3)*.5;
        originalsize(3) = originalsize(3)*.5;
        set(ThisAlgFigureNumber, 'position', originalsize);
    end
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth');
    displaytext = strvcat(['      Image Set # ',num2str(handles.setbeinganalyzed)],...
        ['1 = Saturated       0 = Not Saturated'],...
        [NameImageToCheck1, ':    ', num2str(Saturation1)],...
        [NameImageToCheck2, ':    ', num2str(Saturation2)],...
        [NameImageToCheck3, ':    ', num2str(Saturation3)],...
        ['      '],...
        ['      '],...
        ['FocusScore:'],...
        [NameImageToCheck1, ':    ', num2str(FocusScore1)],...
        [NameImageToCheck2, ':    ', num2str(FocusScore2)],...
        [NameImageToCheck3, ':    ', num2str(FocusScore3)]);
%         [NameImageToCheck4, ':    ', num2str(Saturation4)],...
%         [NameImageToCheck5, ':    ', num2str(Saturation5)],...
%         [NameImageToCheck6, ':    ', num2str(Saturation6)],...
%         [NameImageToCheck4, ':    ', num2str(FocusScore4)],...
%         [NameImageToCheck5, ':    ', num2str(FocusScore5)],...
%         [NameImageToCheck6, ':    ', num2str(FocusScore6)]);
    set(displaytexthandle,'string',displaytext)
end
% 
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Saturation Check module: 
%%%%% .
