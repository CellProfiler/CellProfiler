function handles = MeasureNeighbors(handles)

% Help for the Measure Neighbors module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module determines how many neighbors each object has.
%
% How it works:
% Retrieves a segmented image, in label matrix format, 
%
%
% See also <nothing relevant>.

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
% $Revision$

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.
%
% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the module.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this module with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown. 
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);


%textVAR01 = What did you call the objects whose neighbors you want to measure?
%defaultVAR01 = Cells
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%textVAR02 = Objects are considered neighbors if they are within this distance (pixels), or type 0 to find neighbors if each object were expanded until it touches others:
%defaultVAR02 = 0
NeighborDistance = str2num(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['Segmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline,fieldname)
    error(['Image processing has been canceled. Prior to running the Measure Neighbors module, you must have previously run a segmentation module.  You specified in the MeasureNeighbors module that the desired image was named ', IncomingLabelMatrixImageName(10:end), ', the Measure Neighbors module cannot locate this image.']);
end
IncomingLabelMatrixImage = handles.Pipeline.(fieldname);


if NeighborDistance == 0
    %%% The objects are thickened until they are one pixel shy of
    %%% being 8-connected.  This also turns the image binary rather
    %%% than a label matrix.
    ThickenedBinaryImage = bwmorph(IncomingLabelMatrixImage,'thicken',Inf);
    %%% The objects must be reconverted to a label matrix in a way
    %%% that preserves their prior labeling, so that any measurements
    %%% made on these objects correspond to measurements made by other
    %%% modules.
    ThickenedLabelMatrixImage = bwlabel(ThickenedBinaryImage);
    %%% For each object, one label and one label location is acquired and
    %%% stored.
    [LabelsUsed,LabelLocations] = unique(IncomingLabelMatrixImage);
    %%% The +1 increment accounts for the fact that there are zeros in the
    %%% image, while the LabelsUsed starts at 1.
    LabelsUsed(ThickenedLabelMatrixImage(LabelLocations(2:end))+1) = IncomingLabelMatrixImage(LabelLocations(2:end));
    FinalLabelMatrixImage = LabelsUsed(ThickenedLabelMatrixImage+1);
    IncomingLabelMatrixImage = FinalLabelMatrixImage;
    %%% The NeighborDistance is then set so that neighbors almost
    %%% 8-connected by the previous step are counted as neighbors.
    NeighborDistance = 4;
end



    if sum(sum(IncomingLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage = label2rgb(IncomingLabelMatrixImage,'jet', 'k', 'shuffle');
    else  ColoredLabelMatrixImage = IncomingLabelMatrixImage;
    end




if isempty(str2num(handles.Settings.VariableValues{CurrentModuleNum,2}))
    error('No distance value specified in the Measure Neighbors module')
end
d = max(2,str2num(handles.Settings.VariableValues{CurrentModuleNum,2})+1);

[sr,sc] = size(IncomingLabelMatrixImage);
ImOfNeighbors = -ones(sr,sc);

se = strel('disk',d,0);                   
for k = 1:max(IncomingLabelMatrixImage(:))

    % Cut patch
    [r,c] = find(IncomingLabelMatrixImage == k);
    rmax = min(sr,max(r) + (d+1));
    rmin = max(1,min(r) - (d+1));
    cmax = min(sr,max(c) + (d+1));
    cmin = max(1,min(c) - (d+1));
    p = IncomingLabelMatrixImage(rmin:rmax,cmin:cmax);

    % Extend cell boundary
    pextended = imdilate(p==k,se,'same');
    overlap = p.*pextended;
    NrOfNeighbors = length(setdiff(unique(overlap(:)),0))-1;

    ImOfNeighbors(sub2ind([sr sc],r,c)) = NrOfNeighbors; 
end

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber)
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == 1
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    figure(ThisModuleFigureNumber);
    subplot(2,1,2)
    imagesc(ImOfNeighbors),axis image
    colormap([0 0 0;jet(max(ImOfNeighbors(:)))]);
    colorbar('SouthOutside','FontSize',8)
    set(gca,'FontSize',8)
    title('Cells colored according to the number of neighbors')
    subplot(2,1,1)
    imagesc(ColoredLabelMatrixImage)
    title('Cells colored according to their original colors')
end
