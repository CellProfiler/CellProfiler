function [v, z, OrderedUniqueDoses, OrderedAverageValues] = CP_VZfactors(xcol, ymatr)
% xcol is (Nobservations,1) column vector of grouping values
% (in terms of dose curve it may be Dose).
% ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to observations
% and columns corresponds to different measures.
% v, z are (1, Nmeasures) row vectors containing V- and Z-factors
% for the corresponding measures.
%
% When ranges are zero, we set the V and Z' factor to a very negative
% value.

% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net

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
OrderedUniqueDoses = xs;
OrderedAverageValues = avers;

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