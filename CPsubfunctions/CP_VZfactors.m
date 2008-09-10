function [v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues] = CP_VZfactors(xcol, ymatr)
% xcol is (Nobservations,1) column vector of grouping values
% (in terms of dose curve it may be Dose).
% ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to 
% observations and columns corresponds to different measures.
% v, z and z_bwtn_mean are (1, Nmeasures) row vectors containing V-, Z- and 
% between-mean Z-factors for the corresponding measures.
%
% The one-tailed Z' factor is an attempt to overcome the limitation of the
% Z'-factor formulation used upon populations with moderate or high amounts
% of skewness. In these cases, the tails opposite to the mid-range point
% may lead to a high standard deviation for either population. This will 
% give a low Z' factor even though the population means and samples between
% the means are well-separated. Therefore, the one-tailed Z'factor is 
% calculated with the same formula but using only those samples that lie 
% between the population means. 
% 
% NOTE: The statistical robustness of the one-tailed Z' factor has not been
% determined, and hence should probably not be used at this time.
%
% When ranges are zero, we set the V and Z' factors to a very negative
% value.
%
% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net

% $Revision$

[xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr);
vrange = max(avers) - min(avers);
vstd(vrange == 0) = 1;
vrange(vrange == 0) = 0.000001;
vstd = mean(stds);
v = 1 - 6 .* (vstd ./ vrange);

% Z factor is defined by the positive and negative controls, so we take the
% extremes BY DOSE of the averages and stdevs.
zrange = abs(avers(1, :) - avers(length(xs), :));
zstd = stds(1, :) + stds(length(xs), :);
zstd(zrange == 0) = 1;
zrange(zrange == 0) = 0.000001;
z = 1 - 3 .* (zstd ./ zrange);

% The one-tailed Z factor is defined by using only the samples between the
% means, again defined by DOSE extremes
zrange = abs(avers(1, :) - avers(length(xs), :));
exp1_vals = ymatr(xcol == xs(1),:);
exp2_vals = ymatr(xcol == xs(end),:);
sort_avers = sort([avers(1,:); avers(length(xs),:)],1,'ascend');
stds = zeros(2,size(sort_avers,2));
warning('off','MATLAB:divideByZero');
for i = 1:size(sort_avers,2),
    % Here the std must be calculated using the full formula
    vals1 = exp1_vals(exp1_vals >= sort_avers(1,i) & exp1_vals <= sort_avers(2,i));
    vals2 = exp2_vals(exp2_vals >= sort_avers(1,i) & exp2_vals <= sort_avers(2,i));
    exp1_std = sqrt(sum((vals1 - sort_avers(1,i)).^2)/length(vals1));
    exp2_std = sqrt(sum((vals2 - sort_avers(2,i)).^2)/length(vals2));
    stds(:,i) = cat(1, exp1_std, exp2_std);
end
warning('on','MATLAB:divideByZero');
zstd = stds(1, :) + stds(end, :);

% If means aren't the same and stdev aren't NaN, calculate the value
warning('off','MATLAB:divideByZero');
z_one_tailed = 1 - 3 .* (zstd ./ zrange);
warning('on','MATLAB:divideByZero');
% Otherwise, set it to a really negative value
z_one_tailed(~isfinite(zstd) | zrange == 0) = -1e5;

OrderedUniqueDoses = xs;
OrderedAverageValues = avers;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - LocShrinkMeanStd
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr)
ncols = size(ymatr,2);
[labels, labnum, xs] = LocVectorLabels(xcol);
avers = zeros(labnum, ncols);
stds = avers;
for ilab = 1 : labnum
    labinds = find(labels == ilab);
    labmatr = ymatr(labinds,:);
    if size(labmatr,1) == 1
        for j = 1:size(labmatr,2)
            avers(ilab,j) = labmatr(j);
        end
    else
        avers(ilab, :) = mean(labmatr);
        stds(ilab, :) = std(labmatr, 1);
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - LocVectorLabels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [labels, labnum, uniqsortvals] = LocVectorLabels(x)
n = length(x);
labels = zeros(1, n);
[srt, inds] = sort(x);
prev = srt(1) - 1; % absent value
labnum = 0;
uniqsortvals = labels;
for i = 1 : n
    nextval = srt(i);
    if (nextval ~= prev) % 1-st time for sure
        prev = srt(i);
        labnum = labnum + 1;
        uniqsortvals(labnum) = nextval;
    end
    labels(inds(i)) = labnum;
end
uniqsortvals = uniqsortvals(1 : labnum);
