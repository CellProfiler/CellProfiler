function handles = AlgSaturationCheck1(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR1 = What did you call the image you want to check for saturation?
%defaultVAR1 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
NameImageToCheck1 = handles.(fieldname);
%textVAR2 = What did you call the image you want to check for saturation?
%defaultVAR2 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
NameImageToCheck2 = handles.(fieldname);
%textVAR3 = What did you call the image you want to check for saturation?
%defaultVAR3 = OrigRed
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
NameImageToCheck3 = handles.(fieldname);
%textVAR4 = What did you call the image you want to check for saturation?
%defaultVAR4 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
NameImageToCheck4 = handles.(fieldname);
%textVAR5 = What did you call the image you want to check for saturation?
%defaultVAR5 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
NameImageToCheck5 = handles.(fieldname);
%textVAR6 = What did you call the image you want to check for saturation?
%defaultVAR6 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
NameImageToCheck6 = handles.(fieldname);
%textVAR8 =  For unused colors, leave "N" in the boxes above.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Read (open) the images you want to analyze and assign them to
%%% variables.
if strcmp(upper(NameImageToCheck1), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck1];
    %%% Check whether the image to be analyzed exists in the handles structure.
    if isfield(handles, fieldname) == 0
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck1, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Read the image.
    ImageToCheck1 = handles.(fieldname);
    % figure, imshow(ImageToCheck1), title('ImageToCheck1')
    MaximumPixelValue = max(max(ImageToCheck1));
    if MaximumPixelValue == 1
        Saturation1 = 1;
    else Saturation1 = 0;
    end
    %%% Save the Saturation recording to the handles structure.  The field is named 
    %%% appropriately based on the user's input, with the 'dMT' prefix added so
    %%% that this field will be deleted at the end of the analysis batch.
    fieldname = ['dMTSaturation', NameImageToCheck1];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation1};
    %%% Removed for parallel: guidata(gcbo, handles);
end

if strcmp(upper(NameImageToCheck2), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck2];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck2, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    ImageToCheck2 = handles.(fieldname);
    % figure, imshow(ImageToCheck2), title('ImageToCheck2')
    MaximumPixelValue = max(max(ImageToCheck2));
    if MaximumPixelValue == 1
        Saturation2 = 1;
    else Saturation2 = 0;
    end
    fieldname = ['dMTSaturation', NameImageToCheck2];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation2};
    %%% Removed for parallel: guidata(gcbo, handles);
end        

if strcmp(upper(NameImageToCheck3), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck3];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck3, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    ImageToCheck3 = handles.(fieldname);
    % figure, imshow(ImageToCheck3), title('ImageToCheck3')
    MaximumPixelValue = max(max(ImageToCheck3));
    if MaximumPixelValue == 1;
        Saturation3 = 1;
    else Saturation3 = 0;
    end
    fieldname = ['dMTSaturation', NameImageToCheck3];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation3};
    %%% Removed for parallel: guidata(gcbo, handles);
end        

if strcmp(upper(NameImageToCheck4), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck4];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck4, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    ImageToCheck4 = handles.(fieldname);
    % figure, imshow(ImageToCheck4), title('ImageToCheck4')
    MaximumPixelValue = max(max(ImageToCheck4));
    if MaximumPixelValue == 1;
        Saturation4 = 1;
    else Saturation4 = 0;
    end
    fieldname = ['dMTSaturation', NameImageToCheck4];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation4};
    %%% Removed for parallel: guidata(gcbo, handles);
end        

if strcmp(upper(NameImageToCheck5), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck5];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck5, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    ImageToCheck5 = handles.(fieldname);
    % figure, imshow(ImageToCheck5), title('ImageToCheck5')
    MaximumPixelValue = max(max(ImageToCheck5));
    if MaximumPixelValue == 1;
        Saturation5 = 1;
    else Saturation5 = 0;
    end
    fieldname = ['dMTSaturation', NameImageToCheck5];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation5};
    %%% Removed for parallel: guidata(gcbo, handles);
end        

if strcmp(upper(NameImageToCheck6), 'N') ~= 1
    fieldname = ['dOT', NameImageToCheck6];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Saturation Check module could not find the input image.  It was supposed to be named ', NameImageToCheck6, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    ImageToCheck6 = handles.(fieldname);
    % figure, imshow(ImageToCheck6), title('ImageToCheck6')
    MaximumPixelValue = max(max(ImageToCheck6));
    if MaximumPixelValue == 1;
        Saturation6 = 1;
    else Saturation6 = 0;
    end
    fieldname = ['dMTSaturation', NameImageToCheck6];
    handles.(fieldname)(handles.setbeinganalyzed) = {Saturation6};
    %%% Removed for parallel: guidata(gcbo, handles);
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
    %%% Note that the number of spaces after each measurement name results in
    %%% the measurement numbers lining up properly when displayed in a fixed
    %%% width font.  Also, it costs less than 0.1 seconds to do all of these
    %%% calculations, so I won't bother to retrieve the already calculated
    %%% means and sums from each measurement's code above.
    %%% Checks whether any objects were found in the image.
    displaytext = strvcat(['      Image Set # ',num2str(handles.setbeinganalyzed)],...
        ['1 = Saturated       0 = Not Saturated'],...
        [NameImageToCheck1, ':    ', num2str(Saturation1)],...
        [NameImageToCheck2, ':    ', num2str(Saturation2)],...
        [NameImageToCheck3, ':    ', num2str(Saturation3)],...
        [NameImageToCheck4, ':    ', num2str(Saturation4)],...
        [NameImageToCheck5, ':    ', num2str(Saturation5)],...
        [NameImageToCheck6, ':    ', num2str(Saturation6)]);
    set(displaytexthandle,'string',displaytext)
end

%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Saturation Check module: 
%%%%% .
