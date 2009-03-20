function handles = Align(handles)

% Help for the Align module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Aligns two or three images relative to each other. Particularly useful to
% align microscopy images acquired from different color channels.
% *************************************************************************
%
% For two or three input images, this module determines the optimal
% alignment among them.  If using Mutual Information (see below), this
% works whether the images are correlated or anti-correlated (bright
% in one = bright in the other, or bright in one = dim in the other).
% The Normalized Cross Correlation option requires that the images
% have matching bright and dark areas.  This is useful when the
% microscope is not perfectly calibrated because, for example, proper
% alignment is necessary for primary objects to be helpful to identify
% secondary objects. 
%
% Some important notes for proper use of this module:
% (1) Regardless of the number of input images, they will all be aligned with
% respect to the first image.
% (2) The images are cropped according to the smallest input image. If the
% images are all the same size, then no cropping is performed
% (3) If an image is aligned, the padded pixels are assigned a fill value of zero.
% (4) The module stores the amount of shift between images as a measurement,
% which can be useful for quality control purposes.
%
% Measured feature:                 Feature Number:
% Xshift_Image1NamevsImage2Name  |       1 (e.g., Xshift_BluevsRed)
% Yshift_Image1NamevsImage2Name  |       2 (e.g., Yshift_BluevsRed)
% Xshift_Image2NamevsImage3Name  |       3 (e.g., Xshift_RedvsGreen)
% Yshift_Image2NamevsImage3Name  |       4 (e.g., Yshift_RedvsGreen)
% The latter two are measured only if three images are aligned.
%
% Settings:
%
% After entering the names of the images to be aligned as well as the 
% aligned image name(s), select the method of alignment. There are two 
% choices, one is based on mutual information while the other is based on 
% the cross correlation. When using the cross correlation method, 
% the second image should serve as a template
% and be smaller than the first image selected.

% NEW SETTINGS FOR pyCP:
% In general,
% (1) Make the settings color order RGB (seems more intuitive than BRG)
% (2) Remove the third image input, but add an "Add another image?" button, 
% which creates unlimited "other" images/channels.
% (3) The general approach of Align is to either:
%   (a) align channels within one cycle, or
%   (b) Load a template image (from a separate LoadSingleImage module) and align
%       all cycles to it.  Additionally, it would be great to automate this to load in 
%       a stack of images, as in an illumination correction, and align all images to the 
%       "median aligned" one, though this would probably be difficult.
%
% e.g.
%1. What is the name of the first image to be aligned? (will be displayed as red)
%2. What do you want to call the aligned first image? (default to "AlignedRed")
%3. What is the name of the second image to be aligned? (will be displayed as green) 
%4. What do you want to call the aligned second image? (default to "AlignedGreen")
%5. Button asking "Add another image?" which will create dialogs like #1&2, which will display as "AlignedBlue"
% (any additional chanels would have to be mapped to a new color, so this may not be worth the trouble)
%6. Use Mutual Information or Normalized Cross Correlation as the alignment method?  
%       If using normalized cross correlation, the second image should be the template and smaller than the first. (DLOGAN -- this seems odd to me, 
%       since it goes against the logic of aligning all the images to the first one input) 
% The MoreImageXName settings are not necessary, as far as I can tell, since the above "Add another image?" should subsume this.

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

% DLogan 2009_03_20: Change the color to Red
%textVAR01 = What is the name of the first image to be aligned? (will be displayed as blue) 
%infotypeVAR01 = imagegroup
Image1Name = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
%infotypeVAR02 = imagegroup indep
AlignedImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What is the name of the second image to be aligned? (will be displayed as green) 
%infotypeVAR03 = imagegroup
Image2Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the aligned second image?
%defaultVAR04 = AlignedGreen
%infotypeVAR04 = imagegroup indep
AlignedImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What is the name of the third image to be aligned? (will be displayed as red) 
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
Image3Name = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What do you want to call the aligned third image?
%defaultVAR06 = Do not use
%infotypeVAR06 = imagegroup indep
AlignedImage3Name = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Should this module use Mutual Information or Normalized Cross Correlation to align the images?  If using normalized cross correlation, the second image should be the template and smaller than the first.
%choiceVAR07 = Mutual Information
%choiceVAR07 = Normalized Cross Correlation
AlignMethod = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = (Two image alignment only): If you aligned an image or a sequence with a template, to what other image/sequence do you want to apply the shift calculated above?
%choiceVAR08 = Do not use
%infotypeVAR08 = imagegroup
MoreImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = What do you want to call the subsequently aligned first image?
%defaultVAR09 = Do not use
%infotypeVAR09 = imagegroup indep
MoreAlignedImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = (Two image alignment only): If you aligned an image or a sequence with a template, to what other image/sequence do you want to apply the shift calculated above?
%choiceVAR10 = Do not use
%infotypeVAR10 = imagegroup
MoreImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = What do you want to call the subsequently aligned second image?
%defaultVAR11 = Do not use
%infotypeVAR11 = imagegroup indep
MoreAlignedImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Checks whether the user has chosen "Do not use" in improper places.
if strcmpi(Image1Name,'Do not use') || strcmpi(Image2Name,'Do not use') || strcmpi(AlignedImage1Name,'Do not use') || strcmpi(AlignedImage2Name,'Do not use')
    error(['Image processing was canceled in the ', ModuleName, ' module because you must choose two images to align and name the resulting aligned images - one of the first two images you specified is currently called "Do not use".']);
end
    
if strcmpi(Image3Name,'Do not use') ~= strcmpi(AlignedImage3Name,'Do not use') 
   % If there is a mismatch between the input/output names for image 3:
   % that is, if one is called Do not use but the other is not.
   error(['Image processing was canceled in the ', ModuleName, ' module because you have specified a name for the third image but also marked it "Do not use".']);
end

% Are there three input images? If so, set a flag
AreThereThreeInputImages = ~strcmpi(Image3Name,'Do not use');

% Reads the images
Image1 = CPretrieveimage(handles,Image1Name,ModuleName,'DontCheckColor','CheckScale');
[M1 N1 P1] = size(Image1);
Image2 = CPretrieveimage(handles,Image2Name,ModuleName,'DontCheckColor','CheckScale');
[M2 N2 P2] = size(Image2);
% Same for Image 3.
if AreThereThreeInputImages
    Image3 = CPretrieveimage(handles,Image3Name,ModuleName,'DontCheckColor','CheckScale');
    [M3 N3 P3] = size(Image3);
end
% Same for More Image 1,2.
if ~strcmpi(MoreImage1Name,'Do not use')
    MoreImage1 = CPretrieveimage(handles,MoreImage1Name,ModuleName,'DontCheckColor','CheckScale');
    [M3 N3 P3] = size(MoreImage1);
end
if ~strcmpi(MoreImage2Name,'Do not use')
    MoreImage2 = CPretrieveimage(handles,MoreImage2Name,ModuleName,'DontCheckColor','CheckScale');
    [M3 N3 P3] = size(MoreImage2);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Are all images are the same size? If not take the minimum size and warn the user

if AreThereThreeInputImages, 
    M = [M1 M2 M3];
    N = [N1 N2 N3];
    P = [P1 P2 P3]; 
else
    M = [M1 M2];
    N = [N1 N2];
    P = [P1 P2];
end

Mmin = min(M);
Nmin = min(N);
Pmin = min(P);
if any(diff(M)) || any(diff(N)) || any(diff(P))
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 3 Images not all same size']))
        CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are not all the same size. The images will be cropped to the minimum dimension of (' num2str(Mmin) ', ' num2str(Nmin) ', ' num2str(Pmin) ').'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 3 Images not all same size'],'replace');
    end    
end
CroppedImage1 = Image1(1:Mmin,1:Nmin,1:Pmin);
CroppedImage2 = Image2(1:Mmin,1:Nmin,1:Pmin);
if AreThereThreeInputImages, CroppedImage3 = Image3(1:Mmin,1:Nmin,1:Pmin); end 
    
% If there are color images, let the user know that the color image will be
% converted into grayscale for alignment
if any(P > 1)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images']))
        CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are color. The images will be converted to Gray for alignment and the alignment will be applied to the color images.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images'],'replace');
    end
end
    
% Aligns images 1 and 2 (see subfunctions at the end of the module). The 
% nice thing about imtransform is that it works on 2-D and 3-D images 
% (i.e., grayscale and RGB) with the same arguments
[tx12, ty12] = autoalign(   sum(CroppedImage1,3)/sum(max(max(CroppedImage1))), ...
                            sum(CroppedImage2,3)/sum(max(max(CroppedImage2))), AlignMethod);
Results = ['(Image 1 vs 2: DX ', num2str(tx12), ', DY ', num2str(ty12),')'];

tform = maketform('affine',[1 0 ; 0 1; -tx12 -ty12]);
AlignedImage1 = CroppedImage1;
AlignedImage2 = imtransform(CroppedImage2,tform,'xdata',[1 size(CroppedImage2,2)],'ydata',[1 size(CroppedImage2,1)]);

% If there is a 3rd input image, align image 3 with the newly-aligned image 2
if AreThereThreeInputImages, 
    NotYetAlignedImage3 = imtransform(Image3,tform,'xdata',[1 size(CroppedImage3,2)],'ydata',[1 size(CroppedImage3,1)]);

    [tx23, ty23] = autoalign(   sum(AlignedImage2,3)/sum(max(max(AlignedImage2))), ...
                                sum(NotYetAlignedImage3,3)/sum(max(max(NotYetAlignedImage3))), AlignMethod);
    Results = [ '(Image 1 vs 2: DX ', num2str(tx23),', DY ', num2str(ty23), ') ',...
                '(Image 2 vs 3: DX ', num2str(tx23),', DY ', num2str(ty23),')'];
    tform = maketform('affine',[1 0 ; 0 1; tx23 ty23]);
    AlignedImage3 = imtransform(NotYetAlignedImage3,tform,'xdata',[1 size(CroppedImage3,2)],'ydata',[1 size(CroppedImage3,1)]);
end
    
% Apply this transformation to other images if desired
if ~strcmpi(MoreImage1Name,'Do not use')
    MoreAlignedImage1 = imtransform(MoreImage1,tform,'xdata',[1 size(CroppedImage1,2)],'ydata',[1 size(CroppedImage1,1)]);
end
if ~strcmpi(MoreImage2Name,'Do not use')
    MoreAlignedImage2 = imtransform(MoreImage2,tform,'xdata',[1 size(CroppedImage1,2)],'ydata',[1 size(CroppedImage1,1)]);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Determine the figure number to display the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    % For three input images
    if AreThereThreeInputImages,
        OriginalRGB(:,:,1) = sum(CroppedImage3,3)/sum(max(max(CroppedImage3)));
        OriginalRGB(:,:,2) = sum(CroppedImage2,3)/sum(max(max(CroppedImage2)));
        OriginalRGB(:,:,3) = sum(CroppedImage1,3)/sum(max(max(CroppedImage1)));
        AlignedRGB(:,:,1) = sum(AlignedImage3,3)/sum(max(max(AlignedImage3)));
        AlignedRGB(:,:,2) = sum(AlignedImage2,3)/sum(max(max(AlignedImage2)));
        AlignedRGB(:,:,3) = sum(AlignedImage1,3)/sum(max(max(AlignedImage1)));
    % For two input images
    else
        % Note that the size is recalculated in case images were
        % cropped to be the same size.
        [M1 N1 P1] = size(CroppedImage1);
        OriginalRGB(:,:,1) = zeros(M1,N1);
        OriginalRGB(:,:,2) = sum(CroppedImage2,3)/sum(max(max(CroppedImage2)));
        OriginalRGB(:,:,3) = sum(CroppedImage1,3)/sum(max(max(CroppedImage1)));
        [aM1, aN1, aP1] = size(AlignedImage1);
        AlignedRGB(:,:,1) = zeros(aM1,aN1);
        AlignedRGB(:,:,2) = sum(AlignedImage2,3)/sum(max(max(Image2)));
        AlignedRGB(:,:,3) = sum(AlignedImage1,3)/sum(max(max(Image1)));
    end

    % Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OriginalRGB,'TwoByOne',ThisModuleFigureNumber)
        %%% Add extra space for the text at the bottom.
        Position = get(ThisModuleFigureNumber,'position');
        set(ThisModuleFigureNumber,'position',[Position(1),Position(2)-40,Position(3),Position(4)+40])
    end
    
    hAx = subplot(5,1,1:2,'Parent',ThisModuleFigureNumber);
    CPimagesc(OriginalRGB,handles,hAx);
    title(['Input Images, cycle # ',num2str(handles.Current.SetBeingAnalyzed)],'Parent',hAx);
    % A subplot of the figure window is set to display the adjusted image
    hAx = subplot(5,1,3:4,'Parent',ThisModuleFigureNumber);
    CPimagesc(AlignedRGB,handles,hAx);
    title('Aligned Images','Parent',hAx);

    if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText'))
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [0 0 200 40],'fontname','helvetica','backgroundcolor',[.7 .7 .9],'FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
    end
    set(displaytexthandle,'string',['Offset: ',Results])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Saves the adjusted image to the handles structure so it can be used
% by subsequent modules.
handles.Pipeline.(AlignedImage1Name) = AlignedImage1;
handles.Pipeline.(AlignedImage2Name) = AlignedImage2;
if ~strcmpi(MoreImage1Name,'Do not use')
    handles.Pipeline.(MoreAlignedImage1Name) = MoreAlignedImage1;
end
if ~strcmpi(MoreImage2Name,'Do not use')
    handles.Pipeline.(MoreAlignedImage2Name) = MoreAlignedImage2;
end
if AreThereThreeInputImages
    handles.Pipeline.(AlignedImage3Name) = AlignedImage3;
end

% Stores the shift in alignment as a measurement. We store the image
% names here, because otherwise other Align modules in the pipeline for
% other images would overwrite each other. It *is* still the case two
% Align modules will overwrite each other's measurements if the user
% gives the aligned images all the same names, but we can't catch
% everything.
handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Align',['Xshift_',AlignedImage1Name,'vs',AlignedImage2Name]), tx12);
handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Align',['Yshift_',AlignedImage1Name,'vs',AlignedImage2Name]), ty12);

% If three images were aligned, there are two more measurements to store:
if AreThereThreeInputImages
    handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Align',['Xshift_',AlignedImage2Name,'vs',AlignedImage3Name]), tx23);
    handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Align',['Yshift_',AlignedImage2Name,'vs',AlignedImage3Name]), ty23);
end


%%
%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - autoalign
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [shiftx, shifty] = autoalign(in1, in2, method)
if (strcmp(method, 'Mutual Information')),
    [shiftx, shifty] = autoalign_mutualinf(in1, in2);
else
    [shiftx, shifty] = autoalign_ncc(in1, in2);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - autoalign_ncc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [shiftx, shifty] = autoalign_ncc(in1, in2)
%%% XXX - should check dimensions
ncc = normxcorr2(in1, in2);
[i, j] = find(ncc == max(ncc(:)));
shiftx = j(1) - size(in2, 2);
shifty = i(1) - size(in2, 1);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - autoalign_mutualinf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [shiftx, shifty] = autoalign_mutualinf(in1, in2)
%%% Aligns two images using mutual-information and hill-climbing.
best = mutualinf(in1, in2);
bestx = 0;
besty = 0;
%%% Checks which one-pixel move is best.
for dx=-1:1,
    for dy=-1:1,
        cur = mutualinf(subim(in1, dx, dy), subim(in2, -dx, -dy));
        if (cur > best),
            best = cur;
            bestx = dx;
            besty = dy;
        end
    end
end
if (bestx == 0) && (besty == 0),
    shiftx = 0;
    shifty = 0;
    return;
end
%%% Remembers the lastd direction we moved.
lastdx = bestx;
lastdy = besty;
%%% Loops until things stop improving.
while true,
    [nextx, nexty, newbest] = one_step(in1, in2, bestx, besty, lastdx, lastdy, best);
    if (nextx == 0) && (nexty == 0),
        shiftx = bestx;
        shifty = besty;
        return;
    else
        bestx = bestx + nextx;
        besty = besty + nexty;
        best = newbest;
        lastdx = nextx;
        lastdy = nexty;
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - one_step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nx, ny, nb] = one_step(in1, in2, bx, by, ldx, ldy, best)
%%% Finds the best one pixel move, but only in the same direction(s)
%%% we moved last time (no sense repeating evaluations).  ldx is last
%%% dx, ldy is last dy. 
nb = best;
for dx=-1:1,
    for dy=-1:1,
        if ((dx ~= 0) && (dx == ldx)) || ((dy ~= 0) && (dy == ldy)),
            cur = mutualinf(subim(in1, bx+dx, by+dy), subim(in2, -(bx+dx), -(by+dy)));
            if (cur > nb),
                nb = cur;
                nx = dx;
                ny = dy;
            end
        end
    end
end
if (best == nb),
    %%% no change, so quit searching
    nx = 0;
    ny = 0;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - subim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sub = subim(im, dx, dy)
%%% Subimage with positive or negative offsets
if (dx > 0),
    sub = im(:,dx+1:end);
else
    sub = im(:,1:end+dx);
end
if (dy > 0),
    sub = sub(dy+1:end,:);
else
    sub = sub(1:end+dy,:);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - entropy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function H = entropy(X)

%%% Entropy of samples X
S = imhist(X,256);
%%% if S is probability distribution function N is 1
N=sum(sum(S));
if ((N>0) && (min(S(:))>=0))
    Snz=nonzeros(S);
    H=log2(N)-sum(Snz.*log2(Snz))/N;
else
    H=0;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - entropy2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function H = entropy2(X,Y)

%%% joint entropy of paired samples X and Y Makes sure images are binned to
%%% 256 graylevels
X = double(im2uint8(X));
Y = double(im2uint8(Y));
%%% Creates a combination image of X and Y
XY = 256*X + Y;
S = histc(XY(:),0:(256*256-1));
%%% If S is probability distribution function N is 1
N=sum(sum(S));
if ((N>0) && (min(S(:))>=0))
    Snz=nonzeros(S);
    H=log2(N)-sum(Snz.*log2(Snz))/N;
else
    H=0;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - mutualinf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = mutualinf(X, Y)

%%% Mutual information of images X and Y
I = entropy(X) + entropy(Y) - entropy2(X,Y);