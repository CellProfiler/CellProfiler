function handles = MeasureNeurons(handles,varargin)

% Help for the MeasureNeurons module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module will measure branching info of skelton objects from seed points.
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

%textVAR01 = What did you call the seed objects (e.g. soma)?
%infotypeVAR01 = objectgroup
SeedObjects = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the skeleton image? (This is usually a dendrite or axon image saved from the Morph module)
%infotypeVAR02 = imagegroup
SkeletonName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%%%VariableRevisionNumber = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the incoming skeleton image and assigns it to a variable.
SkeletonImg = CPretrieveimage(handles,SkeletonName,ModuleName,'MustBeBinary','CheckScale');

%%% Reads (opens) the incoming image and assigns it to a variable.
SeedObjectsLabelMatrix = CPretrieveimage(handles,['Segmented', SeedObjects],ModuleName,'DontCheckColor','DontCheckScale',size(SkeletonImg));


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% create new skeleton with 'holes' at nuclei
combined_skel=or(SkeletonImg,SeedObjectsLabelMatrix);
seed_center=imerode(SeedObjectsLabelMatrix,strel('disk',2));
combined_skel=xor(combined_skel,seed_center);
% seed_edge=xor(SeedObjectsLabelMatrix,seed_center);

%% thin a second time to avoid faulty branch points
combined_skel=bwmorph(combined_skel,'thin',Inf);

[IgnoreLabels, DistanceMap] = IdentifySecPropagateSubfunction(SeedObjectsLabelMatrix, zeros(size(SeedObjectsLabelMatrix)), max(combined_skel,SeedObjectsLabelMatrix)> 0, 1.0);

%% Change background from white -> black
DistanceMap(isinf(DistanceMap)) = -Inf; 
branch_points = bwmorph(combined_skel,'branchpoints');

%% Remove branch points that fall on objects removed by
% IdentifySecPropagateSubfunction and thus appear orphaned
branch_points = and(IgnoreLabels,branch_points);

%% transfer labels to branch point image
num_seeds=max(SeedObjectsLabelMatrix(:));
% lab_branch_map=max(SkeletonImg,SeedObjectsLabelMatrix).*branch_points; %% original code
lab_branch_map=max(IgnoreLabels,SeedObjectsLabelMatrix).*branch_points;

NumTrunks = [];
NumNonTrunkBranches = [];
for i=num_seeds:-1:1
    branch_dist_tab{i}=DistanceMap(lab_branch_map==i);
    NumTrunks(i) = sum(branch_dist_tab{i} == 0);
    NumNonTrunkBranches(i) = sum(branch_dist_tab{i} > 0);
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%

trunks = branch_points & (DistanceMap <=1);
nonZeroBranches = branch_points & (DistanceMap > 1);

% Skeleton is blue
r = zeros(size(combined_skel));
g = zeros(size(combined_skel));
b = combined_skel;

%% "Trunk" branchpoints (Distance = 0 branchpoints, i.e. those that fall on the seed object) 
%% are Red
r(trunks) = 1;
g(trunks) = 0;
b(trunks) = 0;

%% NonTrunk branchpoints (Distance > 0 branchpoints, i.e. those that fall beyond the seed object) 
%% are Green
r(nonZeroBranches) = 0;
g(nonZeroBranches) = 1;
b(nonZeroBranches) = 0;

visRGB = cat(3,r,g,b);

drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);

    %%% A subplot of the figure window is set to display the original
    %%% image.
    [hImage,hAx] = CPimagesc(visRGB,handles,ThisModuleFigureNumber);
    title(hAx,['Branchpoints (Distance=0 are red, other branches green) cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% Saves measurements to the handles structure.
%   handles = CPaddmeasurements(handles,ObjectName,FeatureName,Data);
handles = CPaddmeasurements(handles, SeedObjects, ...
    ['NumberTrunks_' SkeletonName], NumTrunks');
handles = CPaddmeasurements(handles, SeedObjects, ...
    ['NumberNonTrunkBranches_' SkeletonName], NumNonTrunkBranches');

        
%% Save Branchpoint Image
BranchpointLabelSeedSkeletonName = [SeedObjects '_' SkeletonName];
handles = CPaddimages(handles,BranchpointLabelSeedSkeletonName,visRGB);
