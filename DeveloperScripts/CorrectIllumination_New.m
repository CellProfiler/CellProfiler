function handles = CorrectIllumination_New(handles)

% Help for the Correct Illumination New module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Beta
% *************************************************************************
%
% Beta
%
% See also CorrectIllumination_Apply, Smooth

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
% $Revision: 4711 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images to be used to calculate the illumination function?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the illumination function?
%defaultVAR02 = IllumBlue
%infotypeVAR02 = imagegroup indep
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

TotalSamples=500000;

if handles.Current.SetBeingAnalyzed == 1
    %%% Creates the empty variable so it can be retrieved later
    %%% without causing an error on the first image set.
    handles.Pipeline.(['Sampled',ImageName]) = [];
end

subsamp_count = round(TotalSamples/handles.Current.NumberOfImageSets);

if subsamp_count > numel(OrigImage)
    subsamp_count = numel(OrigImage);
end

data = [];

% find the resolution of the images
SZ = size(OrigImage);

image = im2double(OrigImage);

% jitter so that pixel values don't bunch at the integers
% (this probably should happen after taking the logarithm of the images)
image = image + rand(SZ) / 256;

% choose
subsampI = floor(SZ(1) * rand(subsamp_count, 1)) + 1;
subsampJ = floor(SZ(2) * rand(subsamp_count, 1)) + 1;
inds = sub2ind(SZ, subsampI, subsampJ);
data = image(inds);
handles.Pipeline.(['Sampled',ImageName]) = [handles.Pipeline.(['Sampled',ImageName]);[subsampI subsampJ data]];

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    data = handles.Pipeline.(['Sampled',ImageName]);
    handles.Measurements.(['Sampled',ImageName]).DataFeatures={'I' 'J' 'Data'};
    handles.Measurements.(['Sampled',ImageName]).Data=data;
    p = ones(100,1);
    for i = [20 10 5 4 3 2 1]
        p = fitIllum(data(1:i:end,:), 3, p, SZ);
    end

    [J, I] = meshgrid(1:SZ(2), 1:SZ(1));
    fitfun = zeros(SZ);
    for j = 1:length(p)
        basis = illumBases(j, I/SZ(1), J/SZ(2));
        fitfun = fitfun + p(j)*basis;
    end

    fitfun = fitfun';
    fitfun = fitfun/min(min(fitfun));
    handles.Pipeline.(IlluminationImageName)=fitfun;
else
    fitfun = OrigImage;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(fitfun,handles);
    title('Illuminatino Function Image');
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function params = fitIllum(D, idx, start, SZ)
I = D(:,1);
J = D(:,2);
for j = 1:length(start)
    ib = illumBases(j, I/SZ(1), J/SZ(2));
    IB(:,j) = ib;
end

[params, fX, iters] = minimize(start, 'sampleBaseEntropy', 50, D(:,idx), IB)


%%% Although the arguments are called "X, Y" they actually correspond to "I, J" when called above.  
%%% It doesn't really affect things, as long as all values for X,Y are in [0,1]
function B = illumBases(idx, X, Y)

% this turns idx into the order for X and Y (assuming the max order is 10)
xpos = floor((idx-1) / 10) + 1;
ypos = mod((idx-1), 10) + 1;

% stretch X and Y
Xlin = X(:);
Ylin = Y(:);

vals = zeros(length(Xlin),10);
vals(:, xpos) = 1;
stage1 = decasteljau(Xlin, vals);
vals = zeros(length(Ylin),10);
vals(:, ypos) = stage1;
stage2 = decasteljau(Ylin, vals);

% reshape the output to match the input
B = reshape(stage2, size(X));

function D = decasteljau(interps, vals)
if (size(vals, 2) == 1)
    D = vals;
    return;
end
D = linear(decasteljau(interps, vals(:,1:end-1)), decasteljau(interps, vals(:,2:end)), interps);

function L = linear(a, b, t)
L = (a + (b - a) .* t);

function [X, fX, i] = minimize(X, f, length, varargin);

% Minimize a differentiable multivariate function.
%
% Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
%
% where the starting point is given by "X" (D by 1), and the function named in
% the string "f", must return a function value and a vector of partial
% derivatives of f wrt X, the "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad
%
% Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-02-23).

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S=['Linesearch']; else S=['Function evaluation']; end

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
    i = i + (length>0);                                      % count iterations?!

    X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
    if length>0, M = MAX; else M = min(MAX, -length-i); end

    while 1                             % keep extrapolating as long as necessary
        x2 = 0; f2 = f0; d2 = d0; df2 = df0; f3 = f0; df3 = df0;
        success = 0;
        while ~success & M > 0
            try
                M = M - 1; i = i + (length<0);                         % count epochs?!
                [f3 df3] = feval(f, X+x3*s, varargin{:});
                if isnan(f3) | isinf(f3) | any(isnan(df3)+isinf(df3)), error, end
                success = 1;
            catch                                % catch any error which occured in f
                x3 = (x2+x3)/2;                                  % bisect and try again
            end
        end
        if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
        d3 = df3'*s;                                                    % new slope
        if d3 > SIG*d0 | f3 > f0+x3*RHO*d0 | M == 0    % are we done extrapolating?
            break
        end
        x1 = x2; f1 = f2; d1 = d2; df1 = df2;             % move point 2 to point 1
        x2 = x3; f2 = f3; d2 = d3; df2 = df3;             % move point 3 to point 2
        A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
        B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
        x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
        if ~isreal(x3) | isnan(x3) | isinf(x3) | x3 < 0   % num prob or wrong sign?
            x3 = x2*EXT;                                 % extrapolate maximum amount
        elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
            x3 = x2*EXT;                                 % extrapolate maximum amount
        elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
            x3 = x2+INT*(x2-x1);
        end
    end                                                       % end extrapolation

    while (abs(d3) > -SIG*d0 | f3 > f0+x3*RHO*d0) & M > 0    % keep interpolating
        if d3 > 0 | f3 > f0+x3*RHO*d0                          % choose subinterval
            x4 = x3; f4 = f3; d4 = d3; df4 = df3;           % move point 3 to point 4
        else
            x2 = x3; f2 = f3; d2 = d3; df2 = df3;           % move point 3 to point 2
        end
        if f4 > f0
            x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
        else
            A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
            B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
            x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
        end
        if isnan(x3) | isinf(x3)
            x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
        end
        x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
        [f3 df3] = feval(f, X+x3*s, varargin{:});
        if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
        M = M - 1; i = i + (length<0);                             % count epochs?!
        d3 = df3'*s;                                                    % new slope
    end                                                       % end interpolation

    if abs(d3) < -SIG*d0 & f3 < f0+x3*RHO*d0           % if line search succeeded
        X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
        fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
        s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
        df0 = df3;                                               % swap derivatives
        d3 = d0; d0 = df0'*s;
        if d0 > 0                                      % new slope must be negative
            s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
        end
        x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
        ls_failed = 0;                              % this line search did not fail
    else
        X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
        if ls_failed | i > abs(length)          % line search failed twice in a row
            break;                             % or we ran out of time, so we give up
        end
        s = -df0; d0 = -s'*s;                                        % try steepest
        x3 = 1/(1-d0);
        ls_failed = 1;                                    % this line search failed
    end
end
fprintf('\n');

function [F, DF] = sampleBaseEntropy(params, data, IB)
fitfun = IB * params;

if (any(fitfun <= 0.0))
    F = 1.0;
    return;
end
ndata = data ./ fitfun;
[ndata, reidx] = sort(ndata);

norm = ndata(end-100)-ndata(100);
ndata = log(norm * ndata);

N = length(ndata);

% This should be a user-set parameter, but it should be much less than N
M = 100;

F = sum(log((N/M)*(ndata(101:end) - ndata(1:end-100)))) / N;

DFsub = sample_sub(fitfun, IB, uint32(reidx), ndata) / N;

DF = DFsub';
