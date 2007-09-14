function IlluminationField = CPcalc_illum_corrxn(Samples, Locations, SmoothingRadius, NumberOfComponents, ImageSize)
% function IlluminationField = CPcalc_illum_corrxn(Samples, Locations, SmoothingRadius, NumberOfComponents, ImageSize)
%
% This function calculates the illumination correction images given
% randomly sampled pixels (in SAMPLES, NxK where K is number of
% channels) and their locations within the image (in LOCATIONS, Nx2).
% SmoothingRadius determines how smooth the bias field is
% (essentially, a Gaussian filter applied to the Illumination
% function.  NumberOfComponents determined how many different pixel
% classes there are (e.g., nuclei, cell, background).  ImageSize gives
% the size of the images the data were taken from.
%
% The method is essentially that given in "Adaptive segmentation of
% MRI data", Wells et al., 1996, except we don't have the class
% distributions.  So we mix estimation of the class distributions and
% the bias field in alternating EM steps.

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

%%% drop any zeros from the samples
[row, col] = find(Samples <= 0.0);
Samples(row, :) = [];
Locations(row, :) = [];

%%% log-transform the pixel values
LogSamples = log(Samples);

%%% find the number of channels in the pixels, and the number of samples
NumberOfSamples = size(Samples, 1);
NumberOfChannels = size(Samples, 2);

%%% Create the initial bias field
IlluminationField = repmat(zeros(ImageSize), [1, 1, NumberOfChannels]);

%%% make an initial guess for the class means (for lack of anything
%%% better, space them by quantile in each channel)
SortedLogSamples = sort(LogSamples);
for class = 1:NumberOfComponents,
    offset = round(NumberOfSamples * class / (NumberOfComponents + 1));
    ClassMeans(class, :) = SortedLogSamples(offset, :);
end

%%% And set each covariance to the full data covariance
ClassCovariances = repmat(cov(LogSamples), [1, 1, NumberOfComponents]);

%%% Set the Class Priors to uniform
ClassPriors = ones(NumberOfComponents, 1) / NumberOfComponents;

figure(2);
density_plot(LogSamples(:,1), LogSamples(:,2));hold('on');


%%% Fit the data
for loopcount = 1:150,
    
    %%%%%%%%%%%%%%%%%%%%
    %%% First pass: estimate means/covariances of components using bias field
    %%%%%%%%%%%%%%%%%%%%
    
    %%% Correct data by bias field
    for i = 1:NumberOfChannels,
        Correction = IlluminationField(:, :, i);
        CorrectedSamples(:, i) = LogSamples(:, i) - Correction(sub2ind(ImageSize, Locations(:, 1), Locations(:, 2)));
    end

    %%% Compute weights
    for i = 1:NumberOfComponents,
        Weights(:, i) = ClassPriors(i) * Gaussian(ClassMeans(i, :), ClassCovariances(:, :, i), CorrectedSamples);
    end
    %%% Normalize
    WeightsSum = sum(Weights, 2);
    WeightsSum(WeightsSum == 0) = 1;
    Weights = Weights ./ repmat(WeightsSum, 1, NumberOfComponents);

    %%% Recompute Class Priors
    ClassPriors = sum(Weights) / NumberOfSamples;
    ClassPriors = ClassPriors / sum(ClassPriors)
    
    %%% remove anything that drops below a threshold of 0.01
    ToRemove = find(ClassPriors < 0.01)
    ClassPriors(ToRemove) = [];
    ClassMeans(ToRemove, :) = [];
    ClassCovariances(:, :, ToRemove) = [];
    Weights(:, ToRemove) = [];
    NumberOfComponents = numel(ClassPriors);
    ClassPriors = ClassPriors / sum(ClassPriors);


    %%% Recompute means and covariances
    figure(2);
    density_plot(CorrectedSamples(:,1), CorrectedSamples(:,2));hold('on');
    hold('on')
    for i = 1:NumberOfComponents,
        ClassMeans(i, :) = sum(CorrectedSamples .* repmat(Weights(:, i), 1, NumberOfChannels)) / sum(Weights(:, i));
        % (yes, this next line should use the just updated class means...)
        WeightedDeltas = (CorrectedSamples - repmat(ClassMeans(i, :), NumberOfSamples, 1)) .* repmat(Weights(:, i), 1, NumberOfChannels);
        ClassCovariances(:, :, i) = WeightedDeltas' * WeightedDeltas / sum(Weights(:, i));
        error_ellipse(ClassCovariances(:, :, i), ClassMeans(i, :));
    end
    drawnow
    hold('off')


    %%%%%%%%%%%%%%%%%%%%
    %%% Second pass: estimate bias field using means/covariances
    %%%%%%%%%%%%%%%%%%%%

    %%% Correct data by bias field
    for i = 1:NumberOfChannels,
        Correction = IlluminationField(:, :, i);
        CorrectedSamples(:,i) = LogSamples(:, i) - Correction(sub2ind(ImageSize, Locations(:, 1), Locations(:, 2)));
    end

    %%% Compute weights
    for i = 1:NumberOfComponents,
        Weights(:, i) = ClassPriors(i) * Gaussian(ClassMeans(i, :), ClassCovariances(:, :, i), CorrectedSamples);
    end
    %%% Normalize
    WeightsSum = sum(Weights, 2);
    WeightsSum(WeightsSum == 0) = 1;
    Weights = Weights ./ repmat(WeightsSum, 1, NumberOfComponents);


    %%% The bias field is a smoothed version of the mean residuals.
    %%% (See equation 21 from Wells's paper.)
    %%%
    %%% _HOWEVER_, we don't want to just shift by the weighted
    %%% residuals, but rather by their
    %%% weighted-inverse-covariance-transformed versions (equation 14)
    %%% (Essentially, the residual should be computed as the steepest
    %%% descent of the Gaussian's probability).  Equation 21 also
    %%% includes, in the denominator, a smoothed version of a uniform,
    %%% unit field as a normalizer.  The actual normalization has to
    %%% be done per-channel, due to the cross-talk from the
    %%% weighted-inverse-covariance transform.
    
    %%% Find the weighted residuals.
    WeightedResiduals = zeros(size(CorrectedSamples));
    for i = 1:NumberOfComponents,
        WeightedDeltas = (CorrectedSamples - repmat(ClassMeans(i, :), NumberOfSamples, 1)) .* repmat(Weights(:, i), 1, NumberOfChannels);
        WeightedResiduals = WeightedResiduals + (inv(ClassCovariances(:, :, i)) * WeightedDeltas')';
    end

    Normalizers = zeros(size(CorrectedSamples));
    %%% Find the normalizers.
    for i = 1:NumberOfChannels,
        FixedDeltas = zeros(size(CorrectedSamples));
        FixedDeltas(:, i) = 1;
        WeightedFixedResiduals = zeros(size(CorrectedSamples));
        for j = 1:NumberOfComponents
            WeightedFixedDeltas = FixedDeltas .* repmat(Weights(:, j), 1, NumberOfChannels);
            WeightedFixedResiduals = WeightedFixedResiduals + (inv(ClassCovariances(:, :, j)) * WeightedFixedDeltas')';
        end
        Normalizers(:, i) = WeightedFixedResiduals(:, i);
    end

    %%% Now, we need to smooth the weighted residuals and the
    %%% normalizers.  We have sparse data, so need to use a sparse
    %%% smoother.
    SmoothResiduals = SparseSmoother(Locations, WeightedResiduals, ImageSize, SmoothingRadius);
    SmoothNormalizers = SparseSmoother(Locations, Normalizers, ImageSize, SmoothingRadius);
    SmoothNormalizers(SmoothNormalizers < 0.01) = 0.01;

    %%% The illumination field is computed relative to corrected
    %%% samples, so we adjust the field relative to its previous
    %%% value.
    IlluminationField = IlluminationField + SmoothResiduals ./ SmoothNormalizers;


    figure(3);
    imagesc(IlluminationField(:, :, 1));
    title('red');
    colorbar
    figure(4);
    imagesc(IlluminationField(:, :, 2));
    title('green');
    colorbar
    drawnow
    if any(isnan(IlluminationField(:))) | any(isinf(IlluminationField(:))),
        'foo'
        keyboard
    end
end


function Probs = Gaussian(Mean, Covariance, Data)
deltas = Data - repmat(Mean, size(Data, 1), 1);
Probs = exp(- sum(deltas' .* (inv(Covariance) * deltas')))' / sqrt(det(Covariance));



function SmoothImage = SparseSmoother(Locations, Samples, ImageSize, SmoothingRadius)
NumberOfChannels = size(Samples, 2);

%%% Normalize by pixel coverage
WeightImage = full(sparse(Locations(:, 1), Locations(:, 2), 1, ImageSize(1), ImageSize(2)));
SmoothWeights = Smoother(WeightImage, SmoothingRadius);
SmoothWeights(SmoothWeights == 0) = 1;

%%% Compute smoothed data from samples
SmoothImage = zeros([ImageSize NumberOfChannels]);
for i = 1:NumberOfChannels,
    SingleChannel = full(sparse(Locations(:, 1), Locations(:, 2), Samples(:, i), ImageSize(1), ImageSize(2)));
    SmoothImage(:, :, i) = Smoother(SingleChannel, SmoothingRadius) ./ SmoothWeights;
end


function SmoothedImage = Smoother(Image, SmoothingRadius)
% Note that we can't use the "convert to 16-bit" trick here, because
% we need to preserve zeros at the boundaries.
Filter = fspecial('gaussian', 2*SmoothingRadius, SmoothingRadius);
SmoothedImage = imfilter(Image, Filter, 0);


function density_plot(xvals, yvals)
minx = min(xvals);
maxx = max(xvals);
xedges = linspace(minx, maxx, 100);
xvals = (xvals - minx) / (maxx - minx);

miny = min(yvals);
maxy = max(yvals);
yedges = linspace(miny, maxy, 100);
yvals = (yvals - miny) / (maxy - miny);

counts = full(sparse(round(99 * yvals + 1), round(99 * xvals + 1), 1, 100, 100));
ax = newplot;
surf(xedges, yedges, zeros(numel(xedges), numel(yedges)), counts, 'EdgeColor','none', 'FaceColor','interp');
view(ax,2);
colormap(ax,flipud(gray(256)))
grid(ax,'off');
drawnow
