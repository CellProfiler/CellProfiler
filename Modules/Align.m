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
% secondary objects. The images are cropped appropriately according to
% this alignment, so the final images will be smaller than the
% originals by a few pixels if alignment is necessary.
% 
% Settings:
%
% After entering the names of the images to be aligned as well as the 
% aligned image name(s), choose whether to display the image produced by 
% this module by selecting "yes" in the appropriate menu. Lastly, select 
% the method of alignment. There are two choices, one is based on mutual 
% information while the other is based on the cross correlation. When using
% the cross correlation method, the second image should serve as a template
% and be smaller than the first image selected.

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

%textVAR01 = What did you call the first image to be aligned? (will be displayed as blue) 
%infotypeVAR01 = imagegroup
Image1Name = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
%infotypeVAR02 = imagegroup indep
AlignedImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What did you call the second image to be aligned? (will be displayed as green) 
%infotypeVAR03 = imagegroup
Image2Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the aligned second image?
%defaultVAR04 = AlignedGreen
%infotypeVAR04 = imagegroup indep
AlignedImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What did you call the third image to be aligned? (will be displayed as red) 
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
Image3Name = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What do you want to call the aligned third image?
%defaultVAR06 = Do not use
%infotypeVAR06 = imagegroup indep
AlignedImage3Name = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = This module calculates the alignment shift and stores it as a measurement. Do you want to actually shift the images and crop them to produce the aligned images? 
%choiceVAR07 = Yes
%choiceVAR07 = No
AdjustImage = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Should this module use Mutual Information or Normalized Cross Correlation to align the images?  If using normalized cross correlation, the second image should be the template and smaller than the first.
%choiceVAR08 = Mutual Information
%choiceVAR08 = Normalized Cross Correlation
AlignMethod = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu


%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the images.
%Image1 = CPretrieveimage(handles,Image1Name,ModuleName,'MustBeGray','CheckScale');
Image1 = CPretrieveimage(handles,Image1Name,ModuleName,'DontCheckColor','CheckScale');
[M1 N1 P1] = size(Image1);
%Image2 = CPretrieveimage(handles,Image2Name,ModuleName,'MustBeGray','CheckScale');
Image2 = CPretrieveimage(handles,Image2Name,ModuleName,'DontCheckColor','CheckScale');
[M2 N2 P2] = size(Image2);
%%% Same for Image 3.
if ~strcmpi(Image3Name,'Do not use')
    Image3 = CPretrieveimage(handles,Image3Name,ModuleName,'DontCheckColor','CheckScale');
    [M3 N3 P3] = size(Image3);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Aligns three input images.
if ~strcmpi(Image3Name,'Do not use')
    % Check to make sure all images are the same size
    % if not take the minimum size of them and warn user
    M = [M1 M2 M3];
    N = [N1 N2 N3];
    P = [P1 P2 P3];    
    if any(diff(M))||any(diff(N))||any(diff(P))
        Mmin=min(M);
        Nmin=min(N);
        Pmin=min(P);
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 3 Images not all same size']))
            CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are not all the same size. The images will be cropped to the minimum dimension of (' num2str(Mmin) ', ' num2str(Nmin) ', ' num2str(Pmin) ').'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 3 Images not all same size'],'replace');
        end        
        Image1=Image1(1:Mmin,1:Nmin,1:Pmin);
        Image2=Image2(1:Mmin,1:Nmin,1:Pmin);
        Image3=Image3(1:Mmin,1:Nmin,1:Pmin);        
    end
    %%% Aligns 1 and 2 (see subfunctions at the end of the module).
    [sx, sy] = autoalign(sum(Image1,3)/sum(max(max(Image1))), sum(Image2,3)/sum(max(max(Image2))), AlignMethod);
    if (P1 > 1)||(P2 > 1)||(P3 > 1)
        %% Color Images
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images']))
            CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are color. The images will be converted to Gray for alignment and the alignment will be applied to the color images.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images'],'replace');
        end
        AlignR1 = subim(Image1(:,:,1), sx, sy);
        AlignG1 = subim(Image1(:,:,2), sx, sy);
        AlignB1 = subim(Image1(:,:,3), sx, sy);
        Temp1 = zeros([size(AlignR1) 3]);
        Temp1(:,:,1) = AlignR1;
        Temp1(:,:,2) = AlignG1;
        Temp1(:,:,3) = AlignB1;
        AlignR2 = subim(Image2(:,:,1), -sx, -sy);
        AlignG2 = subim(Image2(:,:,2), -sx, -sy);
        AlignB2 = subim(Image2(:,:,3), -sx, -sy);
        Temp2 = zeros([size(AlignR2) 3]);
        Temp2(:,:,1) = AlignR2;
        Temp2(:,:,2) = AlignG2;
        Temp2(:,:,3) = AlignB2;
        %%% Assumes 3 is stuck to 2.
        AlignR3 = subim(Image3(:,:,1), -sx, -sy);
        AlignG3 = subim(Image3(:,:,2), -sx, -sy);
        AlignB3 = subim(Image3(:,:,3), -sx, -sy);
        Temp3 = zeros([size(AlignR3) 3]);
        Temp3(:,:,1) = AlignR3;
        Temp3(:,:,2) = AlignG3;
        Temp3(:,:,3) = AlignB3;
        %%% Aligns 2 and 3.
%        [sx2, sy2] = autoalign(Temp2, Temp3, AlignMethod);
        [sx2, sy2] = autoalign(sum(Temp2,3)/sum(max(max(Temp2))), sum(Temp3,3)/sum(max(max(Temp3))), AlignMethod);
        Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy), ...
            ') (2 vs 3: X ', num2str(sx2), ', Y ', num2str(sy2),')'];
        if strcmp(AdjustImage,'Yes') == 1
           %AlignedImage2 = subim(Temp2, sx2, sy2);
           AlignR23 = subim(Temp2(:,:,1), sx2, sy2);
           AlignG23 = subim(Temp2(:,:,2), sx2, sy2);
           AlignB23 = subim(Temp2(:,:,3), sx2, sy2);
           AlignedImage2 = zeros([size(AlignR23) 3]);
           AlignedImage2(:,:,1) = AlignR23;
           AlignedImage2(:,:,2) = AlignG23;
           AlignedImage2(:,:,3) = AlignB23;    
            %AlignedImage3 = subim(Temp3, -sx2, -sy2);
           AlignR32 = subim(Temp3(:,:,1), -sx2, -sy2);
           AlignG32 = subim(Temp3(:,:,2), -sx2, -sy2);
           AlignB32 = subim(Temp3(:,:,3), -sx2, -sy2);
           AlignedImage3 = zeros([size(AlignR32) 3]);
           AlignedImage3(:,:,1) = AlignR32;
           AlignedImage3(:,:,2) = AlignG32;
           AlignedImage3(:,:,3) = AlignB32; 
           %%% 1 was already aligned with 2.
           %AlignedImage1 = subim(Temp1, sx2, sy2);
           AlignR12 = subim(Temp1(:,:,1), sx2, sy2);
           AlignG12 = subim(Temp1(:,:,2), sx2, sy2);
           AlignB12 = subim(Temp1(:,:,3), sx2, sy2);
           AlignedImage1 = zeros([size(AlignR12) 3]);
           AlignedImage1(:,:,1) = AlignR12;
           AlignedImage1(:,:,2) = AlignG12;
           AlignedImage1(:,:,3) = AlignB12; 
        end

    else
        %% Gray Images
        Temp1 = subim(Image1, sx, sy);
        Temp2 = subim(Image2, -sx, -sy);
        %%% Assumes 3 is stuck to 2.
        Temp3 = subim(Image3, -sx, -sy);
        %%% Aligns 2 and 3.
        [sx2, sy2] = autoalign(Temp2, Temp3, AlignMethod);
        Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy), ...
            ') (2 vs 3: X ', num2str(sx2), ', Y ', num2str(sy2),')'];
        if strcmp(AdjustImage,'Yes') == 1
            AlignedImage2 = subim(Temp2, sx2, sy2);
            AlignedImage3 = subim(Temp3, -sx2, -sy2);
            %%% 1 was already aligned with 2.
            AlignedImage1 = subim(Temp1, sx2, sy2);
        end
    end
else %%% Aligns two input images.
    M = [M1 M2];
    N = [N1 N2];
    P = [P1 P2];    
    if any(diff(M))||any(diff(N))||any(diff(P))
        Mmin=min(M);
        Nmin=min(N);
        Pmin=min(P);       
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 2 Images not all same size']))
            CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are not all the same size. The images will be cropped to the minimum dimension of (' num2str(Mmin) ', ' num2str(Nmin) ', ' num2str(Pmin) ').'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': 2 Images not all same size'],'replace');
        end
        Image1=Image1(1:Mmin,1:Nmin,1:Pmin);
        Image2=Image2(1:Mmin,1:Nmin,1:Pmin);
    end
    [sx, sy] = autoalign(sum(Image1,3)/sum(max(max(Image1))), sum(Image2,3)/sum(max(max(Image2))), AlignMethod);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy),')'];
    if strcmp(AdjustImage,'Yes') == 1
       if (P1 > 1)||(P2 > 1)
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images']))
                CPwarndlg(['The images loaded into ' ModuleName ' which is number ' num2str(CurrentModuleNum) ' are color. The images will be converted to Gray for alignment and the alignment will be applied to the color images.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': working on color images'],'replace');
            end
             % Apply subim to each individual color element
             AlignR1 = subim(Image1(:,:,1), sx, sy);
             AlignG1 = subim(Image1(:,:,2), sx, sy);
             AlignB1 = subim(Image1(:,:,3), sx, sy);             
             AlignedImage1 = zeros([size(AlignR1) 3]);
             AlignedImage1(:,:,1) = AlignR1;
             AlignedImage1(:,:,2) = AlignG1; 
             AlignedImage1(:,:,3) = AlignB1;
             AlignR2 = subim(Image2(:,:,1), -sx, -sy);
             AlignG2 = subim(Image2(:,:,2), -sx, -sy);
             AlignB2 = subim(Image2(:,:,3), -sx, -sy);             
             AlignedImage2 = zeros([size(AlignR2) 3]);
             AlignedImage2(:,:,1) = AlignR2;
             AlignedImage2(:,:,2) = AlignG2; 
             AlignedImage2(:,:,3) = AlignB2;
       else    
            AlignedImage1 = subim(Image1, sx, sy);
            AlignedImage2 = subim(Image2, -sx, -sy);
       end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    if strcmp(AdjustImage,'Yes')
        %%% For three input images.
        if (~strcmpi(Image3Name,'Do not use') && all(size(Image1) == size(Image2)) && all(size(Image1) == size(Image3))),
            OriginalRGB(:,:,1) = sum(Image3,3)/sum(max(max(Image3)));
            OriginalRGB(:,:,2) = sum(Image2,3)/sum(max(max(Image2)));
            OriginalRGB(:,:,3) = sum(Image1,3)/sum(max(max(Image1)));
            AlignedRGB(:,:,1) = sum(AlignedImage3,3)/sum(max(max(AlignedImage3)));
            AlignedRGB(:,:,2) = sum(AlignedImage2,3)/sum(max(max(AlignedImage2)));
            AlignedRGB(:,:,3) = sum(AlignedImage1,3)/sum(max(max(AlignedImage1)));
        %%% For two input images.
        elseif all(size(Image1) == size(Image2)),
            %%% Note that the size is recalculated in case images were
            %%% cropped to be the same size.
            [M1 N1 P1] = size(Image1);
            OriginalRGB(:,:,1) = zeros(M1,N1);
            OriginalRGB(:,:,2) = sum(Image2,3)/sum(max(max(Image2)));
            OriginalRGB(:,:,3) = sum(Image1,3)/sum(max(max(Image1)));
            [aM1, aN1, aP1] = size(AlignedImage1);
            AlignedRGB(:,:,1) = zeros(aM1,aN1);
            AlignedRGB(:,:,2) = sum(AlignedImage2,3)/sum(max(max(Image2)));
            AlignedRGB(:,:,3) = sum(AlignedImage1,3)/sum(max(max(Image1)));
        else
            OriginalRGB = Image1;
            AlignedRGB = AlignedImage1;
        end
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OriginalRGB,'TwoByOne',ThisModuleFigureNumber)
        %%% Add extra space for the text at the bottom.
        Position = get(ThisModuleFigureNumber,'position');
        set(ThisModuleFigureNumber,'position',[Position(1),Position(2)-40,Position(3),Position(4)+40])
    end
    if strcmp(AdjustImage,'Yes')
        %%% A subplot of the figure window is set to display the original
        %%% image.
        subplot(5,1,1:2);
        CPimagesc(OriginalRGB,handles);
        title(['Input Images, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the adjusted
        %%%  image.
        subplot(5,1,3:4);
        CPimagesc(AlignedRGB,handles);
        title('Aligned Images');
    end
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

if strcmp(AdjustImage,'Yes')
    %%% Saves the adjusted image to the handles structure so it can be used
    %%% by subsequent modules.
    handles.Pipeline.(AlignedImage1Name) = AlignedImage1;
    handles.Pipeline.(AlignedImage2Name) = AlignedImage2;
    if strcmpi(Image3Name,'Do not use') ~= 1
        handles.Pipeline.(AlignedImage3Name) = AlignedImage3;
    end
end

%%% Stores the shift in alignment as a measurement for quality control
%%% purposes.

%%% If three images were aligned:
if ~strcmpi(Image3Name,'Do not use')
    fieldname = ['Align_',AlignedImage1Name,'_',AlignedImage2Name,'_',AlignedImage3Name,'Features'];
    handles.Measurements.Image.(fieldname) = {'ImageXAlign' 'ImageYAlign' 'ImageXAlignFirstTwoImages' 'ImageYAlignFirstTwoImages'};
    fieldname = ['Align_',AlignedImage1Name,'_',AlignedImage2Name,'_',AlignedImage3Name];
    handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed} = [sx sy sx2 sy2];
else
    fieldname = ['Align_',AlignedImage1Name,'_',AlignedImage2Name,'Features'];
    handles.Measurements.Image.(fieldname) = {'ImageXAlign' 'ImageYAlign'};
    fieldname = ['Align_',AlignedImage1Name,'_',AlignedImage2Name];
    handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed} = [sx sy];
end

% fieldname = ['ImageXAlign', AlignedImage1Name,AlignedImage2Name];
% handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sx};
% fieldname = ['ImageYAlign', AlignedImage1Name,AlignedImage2Name];
% handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sy};

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function [shiftx, shifty] = autoalign(in1, in2, method)
if (strcmp(method, 'Mutual Information')==1),
    [shiftx, shifty] = autoalign_mutualinf(in1, in2);
else
    [shiftx, shifty] = autoalign_ncc(in1, in2);
end

function [shiftx, shifty] = autoalign_ncc(in1, in2)
%%% XXX - should check dimensions
ncc = normxcorr2(in2, in1);
[i, j] = find(ncc == max(ncc(:)));
shiftx = j(1) - size(in2, 2);
shifty = i(1) - size(in2, 1);

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
    end
end

function [nx, ny, nb] = one_step(in1, in2, bx, by, ldx, ldy, best)
%%% Finds the best one pixel move, but only in the same direction(s)
%%% we moved last time (no sense repeating evaluations).  ldx is last
%%% dx, ldy is last dy.  Technically, if one of them is zero, the test
%%% should be more restrictive (XXX).
nb = best;
for dx=-1:1,
    for dy=-1:1,
        if (dx == ldx) || (dy == ldy),
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

function I = mutualinf(X, Y)
%%% Mutual information of images X and Y
I = entropy(X) + entropy(Y) - entropy2(X,Y);