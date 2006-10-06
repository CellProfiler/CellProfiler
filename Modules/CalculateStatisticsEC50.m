function handles = CalculateStatistics(handles)

% Help for the Calculate Statistics module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates measures of assay quality (V and Z' factors) for all
% measured features made from images.
% *************************************************************************
%
% The V and Z' factors are statistical measures of assay quality and are
% calculated for each per-cell and per-image measurement that you have made
% in the pipeline. For example, the Z' factor indicates how well-separated
% the positive and negative controls are. Calculating these values by
% placing this module at the end of a pipeline allows you to choose which
% measured features are most powerful for distinguishing positive and
% negative control samples, or for accurately quantifying the assay's
% response to dose. Both Z' and V factors will be calculated for all
% measured values (Intensity, AreaShape, Texture, etc.). These measurements
% can be exported as the "Experiment" set of data.
%
% For both Z' and V factors, the highest possible value (best assay
% quality) = 1 and they can range into negative values (for assays where
% distinguishing between positive and negative controls is difficult or
% impossible). A Z' factor > 0 is potentially screenable; A Z' factor > 0.5
% is considered an excellent assay.
%
% The Z' factor is based only on positive and negative controls. The V
% factor is based on an entire dose-response curve rather than on the
% minimum and maximum responses. When there are only two doses in the assay
% (positive and negative controls only), the V factor will equal the Z'
% factor.
%
% Note that if the standard deviation of a measured feature is zero for a
% particular set of samples (e.g. all the positive controls), the Z' and V
% factors will equal 1 despite the fact that this is not a useful feature
% for the assay. This occurs when you have only one sample at each dose.
% This also occurs for some non-informative measured features, like the
% number of Cytoplasm compartments per Cell which is always equal to 1.
%
% You must load a simple text file with one entry per cycle (using the Load
% Text module) that tells this module either which samples are positive and
% negative controls, or the concentrations of the sample-perturbing reagent
% (e.g., drug dosage). This text file would look something like this:
%
% [For the case where you have only positive or negative controls; in this
% example the first three images are negative controls and the last three
% are positive controls. They need not be labeled 0 and 1; the calculation
% is based on whichever samples have minimum and maximum dose, so it would
% work just as well to use -1 and 1, or indeed any pair of values:]
% DESCRIPTION Doses
% 0
% 0
% 0
% 1
% 1
% 1
%
% [For the case where you have samples of varying doses; using decimal
% values:]
% DESCRIPTION Doses
% .0000001
% .00000003
% .00000001
% .000000003
% .000000001
% (Note that in this examples, the Z' and V factors will be meaningless because
% there is only one sample at the each dose, so the standard deviation of
% measured features at each dose will be zero).
%
% [Another example where you have samples of varying doses; this time using
% exponential notation:]
% DESCRIPTION Doses
% 10^-7
% 10^-7.523
% 10^-8
% 10^-8.523
% 10^-9
%
%
% The reference for Z' factor is: JH Zhang, TD Chung, et al. (1999) "A
% simple statistical parameter for use in evaluation and validation of high
% throughput screening assays." J Biomolecular Screening 4(2): 67-73.
%
% The reference for V factor is: I Ravkin (2004): Poster #P12024 - Quality
% Measures for Imaging-based Cellular Assays. Society for Biomolecular
% Screening Annual Meeting Abstracts. This is likely to be published 
%
% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the grouping values you loaded for each image cycle? See help for details.
%infotypeVAR01 = datagroup
DataName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Would you like to log-transform the grouping values before attempting to fit a sigmoid curve?
%choiceVAR02 = Yes
%choiceVAR02 = No
LogOrLinear = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = To save the plotted dose response data as an interactive figure, enter the filename here (.fig extension will be automatically added):
%defaultVAR03 = Do not save
%infotypeVAR03 = imagegroup indep
FigureName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets

    %%% Get all fieldnames in Measurements
    ObjectFields = fieldnames(handles.Measurements);

    GroupingStrings = handles.Measurements.Image.(DataName);
    %%% Need column vector
    GroupingValues = str2num(char(GroupingStrings'));

    for i = 1:length(ObjectFields)

        ObjectName = char(ObjectFields(i));

        if strcmp(ObjectName,'Results')
            test=eps;
        end
        %%% Filter out Experiment and Image fields
        if ~strcmp(ObjectName,'Experiment')

            try
                %%% Get all fieldnames in Measurements.(ObjectName)
                MeasureFields = fieldnames(handles.Measurements.(ObjectName));
            catch %%% Must have been text field and ObjectName is class 'cell'
                continue
            end

            for j = 1:length(MeasureFields)

                MeasureFeatureName = char(MeasureFields(j));

                if length(MeasureFeatureName) > 7
                    if strcmp(MeasureFeatureName(end-7:end),'Features')

                        %%% Not placed with above if statement since
                        %%% MeasureFeatureName may not be 8 characters long
                        if ~strcmp(MeasureFeatureName(1:8),'Location')

                            if strcmp(MeasureFeatureName,'ModuleErrorFeatures')
                                continue;
                            end


                            %%% Get Features
                            MeasureFeatures = handles.Measurements.(ObjectName).(MeasureFeatureName);

                            %%% Get Measure name
                            MeasureName = MeasureFeatureName(1:end-8);
                            %%% Check for measurements
                            if ~isfield(handles.Measurements.(ObjectName),MeasureName)
                                error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the measurements you specified.']);
                            end

                            Ymatrix = zeros(length(handles.Current.NumberOfImageSets),length(MeasureFeatures));
                            for k = 1:handles.Current.NumberOfImageSets
                                for l = 1:length(MeasureFeatures)
                                    if isempty(handles.Measurements.(ObjectName).(MeasureName){k})
                                        Ymatrix(k,l) = 0;
                                    else
                                        Ymatrix(k,l) = mean(handles.Measurements.(ObjectName).(MeasureName){k}(:,l));
                                    end
                                end
                            end

                            [GroupingValueRows,n] = size(GroupingValues);
                            [YmatrixRows, n] = size(Ymatrix);
                            if GroupingValueRows ~= YmatrixRows
                                CPwarndlg('There was an error in the Calculate Statistics module involving the number of text elements loaded for it.  CellProfiler will proceed but this module will be skipped.');
                                return;
                            else
                                [v, z] = VZfactors(GroupingValues,Ymatrix);
                                ec50stats = CPec50(GroupingValues,Ymatrix,LogOrLinear);
                                ec = ec50stats(:,3);
                            end

                            measurefield = [ObjectName,'Statistics'];
                            featuresfield = [ObjectName,'StatisticsFeatures'];
                            if isfield(handles.Measurements,'Experiment')
                                if isfield(handles.Measurements.Experiment,measurefield)
                                    OldEnd = length(handles.Measurements.Experiment.(featuresfield));
                                else OldEnd = 0;
                                end
                            else OldEnd = 0;
                            end
                            for a = 1:length(z)
                                handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+a) = z(a);
                                handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+length(z)+a) = v(a);
                                handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+2*length(z)+a) = ec(a);
                                handles.Measurements.Experiment.(featuresfield){OldEnd+a} = ['Zfactor_',MeasureName,'_',MeasureFeatures{a}];
                                handles.Measurements.Experiment.(featuresfield){OldEnd+length(z)+a} = ['Vfactor_',MeasureName,'_',MeasureFeatures{a}];
                                handles.Measurements.Experiment.(featuresfield){OldEnd+2*length(z)+a} = ['EC50_',MeasureName,'_',MeasureFeatures{a}];
                            end

                        end
                    end
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% V AND Z SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net

function [v, z] = VZfactors(xcol, ymatr)
% xcol is (Nobservations,1) column vector of grouping values
% (in terms of dose curve it may be Dose).
% ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to observations
% and columns corresponds to different measures.
% v, z are (1, Nmeasures) row vectors containing V- and Z-factors
% for the corresponding measures.
[xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr);
range = max(avers) - min(avers);
cnstns = find(range == 0);
if (length(cnstns) > 0)
    range(cnstns) = 0.000001;
end
vstd = mean(stds);
v = 1 - 6 .* (vstd ./ range);
zstd = stds(1, :) + stds(length(xs), :);
z = 1 - 3 .* (zstd ./ range);


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


%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EC50 SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

function results=CPec50(conc,responses,LogOrLinear)
% EC50 Function to fit a dose-response data to a 4 parameter dose-response
%   curve.
% 
% Requirements: nlinfit function in the Statistics Toolbox
%           and accompanying m.files: calc_init_params.m and sigmoid.m
% Inputs: 1. a 1 dimensional array of drug concentrations
%         2. the corresponding m x n array of responses 
% Algorithm: generate a set of initial coefficients including the Hill
%               coefficient
%            fit the data to the 4 parameter dose-response curve using
%               nonlinear least squares
% Output: a matrix of the 4 parameters
%         results[m,1]=min
%         results[m,2]=max
%         results[m,3]=ec50
%         results[m,4]=Hill coefficient
%
% Copyright 2004 Carlos Evangelista 
% send comments to CCEvangelista@aol.com
% Version 1.0    01/07/2004

%%% If we are using a log-domain set of doses, we have a better chance of
%%% fitting a sigmoid to the curve if the concentrations are
%%% log-transformed.
if strcmpi(LogOrLinear,'Yes')
    conc = log(conc);
end

[m,n]=size(responses);
results=zeros(n,4);
for i=1:n
    response=responses(:,i);
    initial_params=calc_init_params(conc,response);
    % OPTIONS = statset('display','iter');
    %%% Turns off MATLAB-level warnings but saves the previous warning state.
    PreviousWarningState = warning('off', 'all');
    [coeffs,r,J]=nlinfit(conc,response,'CPsigmoid',initial_params);
    nlintool(conc,response,'CPsigmoid',initial_params);
    if ~strcmpi(FigureName,'Do not save')
        try
            FigureHandle = gcf;
            saveas(FigureHandle,[FigureName,num2str(i),'.fig'],'fig');
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the figure could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    end

    %%% Turns MATLAB-level warnings back on.
    warning(PreviousWarningState);
    for j=1:4
        results(i,j)=coeffs(j);
    end
end



function init_params = calc_init_params(x,y)
% This generates the min, max, x value at the mid-y value, and Hill
% coefficient. These values are used for calculating dose responses using
% nlinfit.

%%% Parameters 1&2
init_params(1) = min(y);
init_params(2) = max(y);
%%% Parameter 3
% OLD:  parms(3)=(min(x)+max(x))/2;
YvalueAt50thPercentile = (min(y)+max(y))/2;
PairedValues = [y,x];
DistanceToCentralYValue = abs(PairedValues(:,1) - YvalueAt50thPercentile);
LocationOfNearest = find(DistanceToCentralYValue == min(DistanceToCentralYValue));
XvalueAt50thPercentile = PairedValues(LocationOfNearest(1),2);
init_params(3) = XvalueAt50thPercentile;
%%% Parameter 4
sizey=size(y);
sizex=size(x);
if (y(1)-y(sizey))./(x(2)-x(sizex))>0
    init_params(4)=(y(1)-y(sizey))./(x(2)-x(sizex));
else
    init_params(4)=1;
end

function [beta,r,J] = nlinfit(X,y,model,beta,options)
%NLINFIT Nonlinear least-squares regression.
%   BETA = NLINFIT(X,Y,MODELFUN,BETA0) estimates the coefficients of a
%   nonlinear regression function, using least squares estimation.  Y is a
%   vector of response (dependent variable) values.  Typically, X is a
%   design matrix of predictor (independent variable) values, with one row
%   for each value in Y and one column for each coefficient.  However, X
%   may be any array that MODELFUN is prepared to accept.  MODELFUN is a
%   function, specified using @, that accepts two arguments, a coefficient
%   vector and the array X, and returns a vector of fitted Y values.  BETA0
%   is a vector containing initial values for the coefficients.
%
%   [BETA,R,J] = NLINFIT(X,Y,MODELFUN,BETA0) returns the fitted
%   coefficients BETA, the residuals R, and the Jacobian J of MODELFUN,
%   evaluated at BETA. You can use these outputs with NLPREDCI to produce
%   confidence intervals for predictions, and with NLPARCI to produce
%   confidence intervals for the estimated coefficients.  sum(R.^2)/(N-P),
%   where [N,P] = size(X), is an estimate of the population variance, and
%   inv(J'*J)*sum(R.^2)/(N-P) is an estimate of the covariance matrix of
%   the estimates in BETA.
%
%   [...] = NLINFIT(X,Y,MODELFUN,BETA0,OPTIONS) specifies control parameters
%   for the algorithm used in NLINFIT.  This argument can be created by a
%   call to STATSET.  Applicable STATSET parameters are:
%
%      'MaxIter'     - Maximum number of iterations allowed.  Defaults to 100.
%      'TolFun'      - Termination tolerance on the residual sum of squares.
%                      Defaults to 1e-8.
%      'TolX'        - Termination tolerance on the estimated coefficients
%                      BETA.  Defaults to 1e-8.
%      'Display'     - Level of display output during estimation.  Choices
%                      are 'off' (the default), 'iter', or 'final'.
%      'DerivStep'   - Relative difference used in finite difference gradient
%                      calculation.  May be a scalar, or the same size as
%                      the parameter vector BETA.  Defaults to EPS^(1/3).
%      'FunValCheck' - Check for invalid values, such as NaN or Inf, from
%                      the objective function [ 'off' | 'on' (default) ].
%
%
%   NLINFIT treats NaNs in Y or MODELFUN(BETA0,X) as missing data, and
%   ignores the corresponding observations.
%
%   Examples:
%
%      Use @ to specify MODELFUN:
%         load reaction;
%         beta = nlinfit(reactants,rate,@mymodel,beta);
%
%      where MYMODEL is a MATLAB function such as:
%         function yhat = mymodel(beta, x)
%         yhat = (beta(1)*x(:,2) - x(:,3)/beta(5)) ./ ...
%                        (1+beta(2)*x(:,1)+beta(3)*x(:,2)+beta(4)*x(:,3));
%   
%      For an example of weighted fitting, see the Statistics Toolbox demo
%      "Weighted Nonlinear Regression".
%
%   See also NLPARCI, NLPREDCI, NLINTOOL, STATSET.

%   References:
%      [1] Seber, G.A.F, and Wild, C.J. (1989) Nonlinear Regression, Wiley.

%   NLINFIT can be used to make a weighted fit with known weights:
%
%      load reaction;
%      w = [8 2 1 6 12 9 12 10 10 12 2 10 8]'; % some example known weights
%      ratew = sqrt(w).*rate;
%      mymodelw = @(beta,X) sqrt(w).*mymodel(beta,X);
% 
%      [betaw,residw,Jw] = nlinfit(reactants,ratew,mymodelw,beta);
%      betaciw = nlparci(betaw,residw,Jw);
%      [ratefitw, deltafitw] = nlpredci(@mymodel,reactants,betaw,residw,Jw);
%      rmse = norm(residw) / (length(w)-length(rate))
% 
%   Predict at the observed x values.  However, the prediction band
%   assumes a weight (measurement precision) of 1 at these points.
%
%      [ratepredw, deltapredw] = ...
%            nlpredci(@mymodel,reactants,betaw,residw,Jw,[],[],'observation');

%   Copyright 1993-2005 The MathWorks, Inc.
%   $Revision: 2.22.2.7.2.1 $  $Date: 2005/07/17 06:07:16 $

if nargin < 4
    error('stats:nlinfit:TooFewInputs','NLINFIT requires four input arguments.');
elseif ~isvector(y)
    error('stats:nlinfit:NonVectorY','Requires a vector second input argument.');
end
if nargin < 5
    options = statset('nlinfit');
else
    options = statset(statset('nlinfit'), options);
end

% Check sizes of the model function's outputs while initializing the fitted
% values, residuals, and SSE at the given starting coefficient values.
model = fcnchk(model);
try
    yfit = model(beta,X);
catch
    [errMsg,errID] = lasterr;
    if isa(model, 'inline')
        error('stats:nlinfit:ModelFunctionError',...
             ['The inline model function generated the following ', ...
              'error:\n%s'], errMsg);
    elseif strcmp('MATLAB:UndefinedFunction', errID) ...
                && ~isempty(strfind(errMsg, func2str(model)))
        error('stats:nlinfit:ModelFunctionNotFound',...
              'The model function ''%s'' was not found.', func2str(model));
    else
        error('stats:nlinfit:ModelFunctionError',...
             ['The model function ''%s'' generated the following ', ...
              'error:\n%s'], func2str(model),errMsg);
    end
end
if ~isequal(size(yfit), size(y))
    error('stats:nlinfit:WrongSizeFunOutput', ...
          'MODELFUN should return a vector of fitted values the same length as Y.');
end

% Set up convergence tolerances from options.
maxiter = options.MaxIter;
betatol = options.TolX;
rtol = options.TolFun;
fdiffstep = options.DerivStep;
funValCheck = strcmp(options.FunValCheck, 'on');

% Find NaNs in either the responses or in the fitted values at the starting
% point.  Since X is allowed to be anything, we can't just check for rows
% with NaNs, so checking yhat is the appropriate thing.  Those positions in
% the fit will be ignored as missing values.  NaNs that show up anywhere
% else during iteration will be treated as bad values.
nans = (isnan(y(:)) | isnan(yfit(:))); % a col vector
n = sum(~nans);
p = numel(beta);

sqrteps = sqrt(eps(class(beta)));
J = zeros(n,p,class(yfit));
r = y(:) - yfit(:);
r(nans) = [];
sse = r'*r;
if funValCheck && ~isfinite(sse), checkFunVals(r); end

zbeta = zeros(size(beta),class(beta));
zerosp = zeros(p,1,class(r));

% Set initial weight for LM algorithm.
lambda = .01;

% Set the level of display
switch options.Display
case 'off',    verbose = 0;
case 'notify', verbose = 1;
case 'final',  verbose = 2;
case 'iter',   verbose = 3;
end

if verbose > 2 % iter
    disp(' ');
    disp('                                     Norm of         Norm of');
    disp('   Iteration             SSE        Gradient           Step ');
    disp('  -----------------------------------------------------------');
    disp(sprintf('      %6d    %12g',0,sse));
end

iter = 0;
breakOut = false;
while iter < maxiter
    iter = iter + 1;
    betaold = beta;
    sseold = sse;

    % Compute a finite difference approximation to the Jacobian
    for k = 1:p
        delta = zbeta;
        if (beta(k) == 0)
            nb = sqrt(norm(beta));
            delta(k) = fdiffstep * (nb + (nb==0));
        else
            delta(k) = fdiffstep*beta(k);
        end
        yplus = model(beta+delta,X);
        dy = yplus(:) - yfit(:);
        dy(nans) = [];
        J(:,k) = dy/delta(k);
    end

    % Levenberg-Marquardt step: inv(J'*J+lambda*D)*J'*r
    diagJtJ = sum(abs(J).^2, 1);
    if funValCheck && ~all(isfinite(diagJtJ)), checkFunVals(J(:)); end
    Jplus = [J; diag(sqrt(lambda*diagJtJ))];
    rplus = [r; zerosp];
    step = Jplus \ rplus;
    beta(:) = beta(:) + step;

    % Evaluate the fitted values at the new coefficients and
    % compute the residuals and the SSE.
    yfit = model(beta,X);
    r = y(:) - yfit(:);
    r(nans) = [];
    sse = r'*r;
    if funValCheck && ~isfinite(sse), checkFunVals(r); end

    % If the LM step decreased the SSE, decrease lambda to downweight the
    % steepest descent direction.
    if sse < sseold
        lambda = 0.1*lambda;

    % If the LM step increased the SSE, repeatedly increase lambda to
    % upweight the steepest descent direction and decrease the step size
    % until we get a step that does decrease SSE.
    else
        while sse > sseold
            lambda = 10*lambda;
            if lambda > 1e16
                CPwarndlg(['There was a problem in fitting the curve for EC50 calculations. ',...
                    'Unable to find a step that will decrease SSE.  Returning results from last iteration.'],...
                    'Warning','modal');
                breakOut = true;
                break
            end
            Jplus = [J; diag(sqrt(lambda*sum(J.^2,1)))];
            step = Jplus \ rplus;
            beta(:) = betaold(:) + step;
            yfit = model(beta,X);
            r = y(:) - yfit(:);
            r(nans) = [];
            sse = r'*r;
            if funValCheck && ~isfinite(sse), checkFunVals(r); end
        end
    end
    
    if verbose > 2 % iter
        disp(sprintf('      %6d    %12g    %12g    %12g', ...
                     iter,sse,norm(2*r'*J),norm(step)));
    end

    % Check step size and change in SSE for convergence.
    if norm(step) < betatol*(sqrteps+norm(beta))
        if verbose > 1 % 'final' or 'iter'
            disp('Iterations terminated: relative norm of the current step is less than OPTIONS.TolX');
        end
        break
    elseif abs(sse-sseold) <= rtol*sse
        if verbose > 1 % 'final' or 'iter'
            disp('Iterations terminated: relative change in SSE less than OPTIONS.TolFun');
        end
        break
    elseif breakOut
        break
    end
end

if (iter >= maxiter)
    CPwarndlg(['There was a problem in fitting the curve for EC50 calculations. ',...
        'Iteration limit exceeded.  Returning results from final iteration.'],...
        'Warning','modal');
end

% If the Jacobian is ill-conditioned, then two parameters are probably
% aliased and the estimates will be highly correlated.  Prediction at new x
% values not in the same column space is dubious.  NLPARCI will have
% trouble computing CIs because the inverse of J'*J is difficult to get
% accurately.  NLPREDCI will have the same difficulty, and in addition,
% will in effect end up taking the difference of two very large, but nearly
% equal, variance and covariance terms, lose precision, and so the
% prediction bands will be erratic.
[Q,R] = qr(J,0);
if n <= p
    CPwarndlg(['There was a problem in fitting the curve for EC50 calculations. ',...
        'The model is overparameterized, and model parameters are not ' ,...
        'identifiable.  You will not be able to compute confidence or ' ...
        'prediction intervals, and you should use caution in making predictions.'],...
        'Warning','modal');
elseif condest(R) > 1/(eps(class(beta)))^(1/2)
    CPwarndlg(['There was a problem in fitting the curve for EC50 calculations. ',...
        'The Jacobian at the solution is ill-conditioned, and some ' ...
        'model parameters may not be estimated well (they are not ' ...
        'identifiable). Use caution in making predictions.'],...
        'Warning','modal');
end
if nargout > 1
    % Return residuals and Jacobian that have missing values where needed.
    r = y - yfit;
    JJ(~nans,:) = J;
    JJ(nans,:) = NaN;
    J = JJ;
    
    % We could estimate the population variance and the covariance matrix
    % for beta here as
    % mse = sum(abs(r).^2)/(n-p);
    % Rinv = inv(R);
    % Sigma = Rinv*Rinv'*mse;
end


function checkFunVals(v)
if any(~isfinite(v))
    error('stats:nlinfit:NonFiniteFunOutput', ...
          'MODELFUN has returned Inf or NaN values.');
end

function options = statset(varargin)
%STATSET Create/alter STATS options structure.
%   OPTIONS = STATSET('PARAM1',VALUE1,'PARAM2',VALUE2,...) creates a
%   statistics options structure OPTIONS in which the named parameters have
%   the specified values.  Any unspecified parameters are set to [].  When
%   you pass OPTIONS to a statistics function, a parameter set to []
%   indicates that the function uses its default value for that parameter.
%   Case is ignored for parameter names, and unique partial matches are
%   allowed.  NOTE: For parameters that are string-valued, the complete
%   string is required for the value; if an invalid string is provided, the
%   default is used.
%
%   OPTIONS = STATSET(OLDOPTS,'PARAM1',VALUE1,...) creates a copy of
%   OLDOPTS with the named parameters altered with the specified values.
%
%   OPTIONS = STATSET(OLDOPTS,NEWOPTS) combines an existing options
%   structure OLDOPTS with a new options structure NEWOPTS.  Any parameters
%   in NEWOPTS with non-empty values overwrite the corresponding old
%   parameters in OLDOPTS.
%
%   STATSET with no input arguments and no output arguments displays all
%   parameter names and their possible values, with defaults shown in {}
%   when the default is the same for all functions that use that option.
%   Use STATSET(STATSFUNCTION) (see below) to see function-specific
%   defaults for a specific function.
%
%   OPTIONS = STATSET (with no input arguments) creates an options
%   structure OPTIONS where all the fields are set to [].
%
%   OPTIONS = STATSET(STATSFUNCTION) creates an options structure with all
%   the parameter names and default values relevant to the optimization
%   function named in STATSFUNCTION.  STATSET sets parameters in OPTIONS to
%   [] for parameters that are not valid for STATSFUNCTION.  For example,
%   statset('factoran') or statset(@factoran) returns an options structure
%   containing all the parameter names and default values relevant to the
%   function 'factoran'.
%
%   STATSET parameters:
%      Display     - Level of display [ off | notify | final ]
%      MaxFunEvals - Maximum number of objective function evaluations
%                    allowed [ positive integer ]
%      MaxIter     - Maximum number of iterations allowed [ positive integer ]
%      TolBnd      - Parameter bound tolerance [ positive scalar ]
%      TolFun      - Termination tolerance for the objective function
%                    value [ positive scalar ]
%      TolX        - Termination tolerance for the parameters [ positive scalar ]
%      GradObj     - Objective function can return a gradient vector as a
%                    second output [ off | on ]
%      DerivStep   - Relative difference used in finite difference derivative
%                    calculations.  May be a scalar or the same size as the
%                    parameter vector [ positive scalar or vector ]
%      FunValCheck - Check for invalid values, such as NaN or Inf, from
%                    the objective function [ off | on ]
%
%   See also STATGET.

%   Copyright 1993-2005 The MathWorks, Inc.
%   $Revision: 1.3.6.10 $  $Date: 2005/05/31 16:45:22 $

% Print out possible values of properties.
if (nargin == 0) && (nargout == 0)
    fprintf('                Display: [ off | notify | final ]\n');
    fprintf('            MaxFunEvals: [ positive integer ]\n');
    fprintf('                MaxIter: [ positive integer ]\n');
    fprintf('                 TolBnd: [ positive scalar ]\n');
    fprintf('                 TolFun: [ positive scalar ]\n');
    fprintf('                   TolX: [ positive scalar ]\n')
    fprintf('                GradObj: [ off | on ]\n')
    fprintf('              DerivStep: [ positive scalar or vector ]\n')
    fprintf('            FunValCheck: [ off | on ]\n')
    fprintf('\n');
    return;
end

options = struct('Display', [], 'MaxFunEvals', [], 'MaxIter', [], ...
                 'TolBnd', [], 'TolFun', [], 'TolX', [], 'GradObj', [], ...
                 'DerivStep', [], 'FunValCheck', []);

% If a function name/handle was passed in, then return the defaults.
if nargin == 1
    arg = varargin{1};
    if (ischar(arg) || isa(arg,'function_handle'))
        if isa(arg,'function_handle')
            arg = func2str(arg);
        end
        % Display is off by default.  The individual fitters have their own
        % warning/error messages that can be controlled via IDs.  The
        % optimizers print out text when display is on, but do not generate
        % warnings or errors per se.
        options.Display = 'off';
        switch lower(arg)
        case 'factoran'
            options.MaxFunEvals = 400;
            options.MaxIter = 100;
            options.TolFun = 1e-8;
            options.TolX = 1e-8;
        case {'normfit' 'lognfit' 'gamfit' 'bisafit' 'invgfit' 'logifit'...
              'loglfit' 'nakafit' 'coxphfit'} % these use statsfminbx
            options.MaxFunEvals = 200;
            options.MaxIter = 100;
            options.TolBnd = 1e-6;
            options.TolFun = 1e-8;
            options.TolX = 1e-8;
        case {'evfit' 'wblfit'} % these use fzero (gamfit sometimes does too)
            options.TolX = 1e-6;
        case {'gpfit' 'gevfit' 'nbinfit' 'ricefit' 'tlsfit'} % these use fminsearch
            options.MaxFunEvals = 400;
            options.MaxIter = 200;
            options.TolBnd = 1e-6;
            options.TolFun = 1e-6;
            options.TolX = 1e-6;
        case 'nlinfit'
            options.MaxIter = 200;
            options.TolFun = 1e-8;
            options.TolX = 1e-8;
            options.DerivStep = eps^(1/3);
            options.FunValCheck = 'on';
        case 'mlecustom'
            options.MaxFunEvals = 400;
            options.MaxIter = 200;
            options.TolBnd = 1e-6;
            options.TolFun = 1e-6;
            options.TolX = 1e-6;
            options.GradObj = 'off';
            options.DerivStep = eps^(1/3);
            options.FunValCheck = 'on';
        case 'mlecov'
            options.GradObj = 'off';
            options.DerivStep = eps^(1/4);
        case 'mdscale'
            options.MaxIter = 200;
            options.TolFun = 1e-6;
            options.TolX = 1e-6;
        otherwise
            error('stats:statset:BadFunctionName',...
                  'No default options available for the function ''%s''.',arg);
        end
        return
	end
end

names = fieldnames(options);
lowNames = lower(names);
numNames = numel(names);

% Process OLDOPTS and NEWOPTS, if it's there.
i = 1;
while i <= nargin
    arg = varargin{i};
    % Check if we're into the param name/value pairs yet.
    if ischar(arg), break; end

    if ~isempty(arg) % [] is a valid options argument
        if ~isa(arg,'struct')
            error('stats:statset:BadInput',...
                  ['Expected argument %d to be a parameter name string or ' ...
                   'an options structure\ncreated with STATSET.'], i);
        end
        argNames = fieldnames(arg);
        for j = 1:numNames
            name = names{j};
            if any(strcmp(name,argNames))
                val = arg.(name);
                if ~isempty(val)
                    if ischar(val)
                        val = lower(deblank(val));
                    end
                    [valid, errid, errmsg] = checkparam(name,val);
                    if valid
                        options.(name) = val;
                    elseif ~isempty(errmsg)
                        error(errid,errmsg);
                    end
                end
            end
        end
    end
    i = i + 1;
end

% Done with OLDOPTS and NEWOPTS, now parse parameter name/value pairs.
if rem(nargin-i+1,2) ~= 0
    error('stats:statset:BadInput',...
          'Arguments must occur in name-value pairs.');
end
expectval = false; % start expecting a name, not a value
while i <= nargin
    arg = varargin{i};

    % Process a parameter name.
    if ~expectval
        if ~ischar(arg)
            error('stats:statset:BadParameter',...
                  'Expected argument %d to be a parameter name string.', i);
        end
        lowArg = lower(arg);
        j = strmatch(lowArg,lowNames);
        if numel(j) == 1 % one match
            name = names{j};
        elseif length(j) > 1 % more than one match
            % Check for any exact matches (in case any names are subsets of others)
            k = strmatch(lowArg,lowNames,'exact');
            if numel(k) == 1
                name = names{k};
            else
                matches = names{j(1)};
                for k = j(2:end)', matches = [matches ', ' names{k}]; end
                error('stats:statset:BadParameter',...
                      'Ambiguous parameter name ''%s'' (%s)', arg, matches);
            end
        else %if isempty(j) % no matches
            error('stats:statset:BadParameter',...
                  'Unrecognized parameter name ''%s''.', arg);
        end
        expectval = true; % expect a value next

    % Process a parameter value.
    else
        if ischar(arg)
            arg = lower(deblank(arg));
        end
        [valid, errid, errmsg] = checkparam(name,arg);
        if valid
            options.(name) = arg;
        elseif ~isempty(errmsg)
            error(errid,errmsg);
        end
        expectval = false; % expect a name next
    end
    i = i + 1;
end


%-------------------------------------------------
function [valid, errid, errmsg] = checkparam(name,value)
%CHECKPARAM Validate a STATSET parameter value.
%   [VALID,ID,MSG] = CHECKPARAM('name',VALUE) checks that the specified
%   value VALUE is valid for the parameter 'name'.
valid = true;
errmsg = '';
errid = '';

% Empty is always a valid parameter value.
if isempty(value)
    return
end

switch name
case {'TolFun','TolBnd','TolX'} % positive real scalar
    if ~isfloat(value) || ~isreal(value) || ~isscalar(value) || any(value <= 0)
        valid = false;
        errid = 'stats:statset:BadTolerance';
        if ischar(value)
            errmsg = sprintf('STATSET parameter ''%s'' must be a real positive scalar (not a string).',name);
        else
            errmsg = sprintf('STATSET parameter ''%s'' must be a real positive scalar.',name);
        end
    end
case {'Display'} % off,notify,final,iter
    values = ['off   '; 'notify'; 'final '; 'iter  '];
    if ~ischar(value) || isempty(strmatch(value,values,'exact'))
        valid = false;
        errid = 'stats:statset:BadDisplay';
        errmsg = sprintf('STATSET parameter ''%s'' must be ''off'', ''notify'', ''final'', or ''iter''.',name);
    end
case {'MaxIter' 'MaxFunEvals'} % non-negative integer, possibly inf
    if ~isfloat(value) || ~isreal(value) || ~isscalar(value) || any(value < 0)
        valid = false;
        errid = 'stats:statset:BadMaxValue';
        if ischar(value)
            errmsg = sprintf('STATSET parameter ''%s'' must be a real positive scalar (not a string).',name);
        else
            errmsg = sprintf('STATSET parameter ''%s'' must be a real positive scalar.',name);
        end
    end
case {'GradObj' 'FunValCheck'}
    values = ['off'; 'on '];
    if ~ischar(value) || isempty(strmatch(value,values,'exact'))
        valid = false;
        errid = 'stats:statset:BadFlagValue';
        errmsg = sprintf('STATSET parameter ''%s'' must be ''off'' or ''on''.',name);
    end
case 'DerivStep'
    if ~isfloat(value) || ~isreal(value) || any(value <= 0)
        valid = false;
        errid = 'stats:statset:BadDifference';
        errmsg = sprintf('STATSET parameter ''%s'' must contain real positive values.',name);
    end
otherwise
    valid = false;
    errid = 'stats:statset:BadParameter';
    errmsg = sprintf('Invalid STATSET parameter name: ''%s''.',name);
end

function nlintool(x,y,model,beta0,varargin)
%NLINTOOL Interactive graphical tool for nonlinear fitting and prediction.
%   NLINTOOL(X,Y,FUN,BETA0) is a prediction plot that provides a 
%   nonlinear curve fit to (x,y) data. It plots 95% global confidence 
%   bands for predictions as two red curves. 
%
%   Y is a vector.  X is a vector or matrix with the same number of rows
%   as Y.  FUN is a function that accepts two arguments, a coefficient
%   vector and an array of X values, and returns a vector of fitted Y
%   values.  BETA0 is a vector containing initial guesses for the
%   coefficients.
%
%   NLINTOOL(...,'PARAM1',val1,'PARAM2'val2,...) specifies one or more
%   of the following parameter name/value pairs:
%
%      'alpha'    An ALPHA value for 100(1-ALPHA) percent confidence bands
%      'xname'    Character array of X variable names
%      'yname'    Y variable name
%      'plotdata' Either 'on' to show data on plot, or 'off' (default) for
%                 fit only.  'on' is not allowed with multiple X variables.
%
%   You can drag the dotted white reference line and watch the predicted
%   values update simultaneously.  Alternatively, you can get a specific
%   prediction by typing the "X" value into an editable text field.
%   Use the Export button to move specified variables to the base workspace.
%   Use the Bounds menu to change the type of confidence bounds.
%
%   Examples
%   --------
%   FUN can be specified using @:
%      nlintool(x, y, @myfun, b0)
%   where MYFUN is a MATLAB function such as:
%      function yhat = myfun(beta, x)
%      b1 = beta(1);
%      b2 = beta(2);
%      yhat = 1 ./ (1 + exp(b1 + b2*x));
%
%   FUN can also be an inline object:
%      fun = inline('1 ./ (1 + exp(b(1) + b(2)*x))', 'b', 'x')
%      nlintool(x, y, fun, b0)
%
%   See also NLINFIT, NLPARCI, NLPREDCI.
   
%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision: 2.30.4.6 $  $Date: 2004/07/05 17:02:59 $

if (nargin == 0)
    action = 'start';
elseif ~ischar(x) 
    action = 'start';
else
    action = x;
end

%On recursive calls get all necessary handles and data.
if ~strcmp(action,'start')
   if nargin == 2,
      flag = y;
   end

   %set(0,'showhiddenhandles','off')
   nlin_fig = findobj(0,'Tag','nlinfig');
   
   k = gcf;
   idx = find(nlin_fig == k);
   if isempty(idx)
      return
   end
   nlin_fig = nlin_fig(idx);      
   if strcmp(action,'down')
       set(nlin_fig,'WindowButtonMotionFcn','nlintool(''motion'',1)');
   end
   ud = get(nlin_fig,'Userdata');
   
   R             = ud.R;
   beta          = ud.beta;
   model         = ud.model;
   x_field       = ud.x_field;
   nlin_axes     = ud.nlin_axes;
   xsettings     = ud.xsettings;
   y_field1      = ud.y_field(1);
   y_field2      = ud.y_field(2);
   reference_line= ud.reference_line;
   last_axes     = ud.last_axes;  
   n             = size(x_field,2);

   xrange         = zeros(n,2);
   newx           = zeros(n,1);

   for k = 1:n         
       xrange(k,1:2) = get(nlin_axes(k),'XLim');
   end
   newy = str2double(get(y_field1,'String'));  
   newdeltay = str2double(get(y_field2,'String'));  
end

switch action

case 'start'

if (nargin == 0)
   msg = sprintf(['To use NLINTOOL with your own data, type:\n' ...
                  '   nlintool(x,y,fun,beta0)\n\n' ...
                  'Type "help nlintool" for more information\n\n' ...
                  'Click OK to analyze some sample data.']);
   try
      s = load('reaction');
      x = s.reactants;
      y = s.rate;
      b = s.beta;
   catch
      s = [];
   end
   hmsg = warndlg(msg, 'Nonlinear Fitting');
   if (length(s)>0)
      nlintool(x, y, 'hougen', b);
   end
   figure(hmsg);
   return
end

if (nargin<4)
    error('stats:nlintool:TooFewInputs',...
          'NLINTOOL requires at least four arguments.');
end

if (size(x,1)==1), x = x(:); end
if (size(y,1)==1), y = y(:); end
wasnan = (isnan(y) | any(isnan(x),2));
if (any(wasnan))
   x(wasnan,:) = [];
   y(wasnan)=[];
end

% Defaults for optional inputs
alpha = 0.05;
xname = '';
yname = '';
if size(x,2)>1
   plotdata = 'off';
else
   plotdata = 'on';
end

% Deal with grandfathered syntax:  nlintool(x,y,fun,b0,alpha,xname,yname)
if length(varargin)>0 && isnumeric(varargin{1})
   if length(varargin)>3
       error('stats:nlintool:BadSyntax',...
             'Too many inputs or bad name/value pairs');
   end
   if ~isempty(varargin{1})
       alpha = varargin{1};
   end
   if length(varargin)>=2
       xname = varargin{2};
   end
   if length(varargin)>=3
       yname = varargin{3};
   end
else
   okargs =   {'alpha' 'xname' 'yname' 'plotdata'};
   defaults = {alpha   xname   yname   plotdata};
   [eid emsg alpha xname yname plotdata] = ...
                   statgetargs(okargs,defaults,varargin{:});
   if ~isempty(eid)
      error(['stats:nlintool:' eid], emsg);
   end
end

if ~isequal(plotdata,'off')
   if ~isequal(plotdata,'on')
       error('stats:nlintool:BadPlotData',...
             'PLOTDATA parameter must be ''on'' or ''off''.');
   elseif size(x,2)>1
       warning('stats:nlintool:BadPlotData',...
               'Cannot plot data when there are multiple X variables.');
       plotdata = 'off';
   end
end

if nargin < 6, xname = ''; end
xstr = cell(size(x,2),1);
for k = 1:size(x,2)
   if isempty(xname)
      xstr{k} = ['X' int2str(k)];
   else
      xstr{k} = xname(k,:);
   end
end

if nargin == 7
   ystr = yname;
else
   ystr = 'Predicted Y';
end

% Fit nonlinear model.  Catch any error so we can restore the warning
% state before re-throwing the error.
[lastwmsg,lastwid] = lastwarn;
lastwstate = warning;
lastwarn('');
warning('off');

try
   [ud.beta, residuals, J] = nlinfit(x,y,model,beta0);
   ok = true;
catch
   ok = false;
end

wmsg = lastwarn;
lastwarn(lastwmsg,lastwid);
warning(lastwstate);
if ~ok
   rethrow(lasterror)
end

if ~isempty(wmsg)
   hwarn = warndlg(wmsg);
else
   hwarn = [];
end

yhat = feval(model,ud.beta,x);

p = length(ud.beta);
df = max(size(y)) - p;

ud.residuals = residuals;
ud.rmse = sqrt(sum(residuals.*residuals)./df);
ud.crit_f = sqrt(p*finv(1 - alpha,p,df))*ud.rmse;
ud.crit_t = tinv(1 - alpha/2,df)*ud.rmse;
ud.crit = ud.crit_f;
ud.model = model;   
ud.simflag = 1;        % simultaneous confidence intervals
ud.obsflag = 0;        % not for new observation (for mean)
ud.confflag = 1;       % show confidence bounds

[Q,R] = qr(J,0);
ud.R = R;

% Set positions of graphic objects
maxx = max(x);
minx = min(x);
xrange = maxx - minx;
xlims = [minx - 0.025 .* xrange; maxx + 0.025 .* xrange]';

[m,n]          = size(x);

nlin_axes      = zeros(n,1);
fitline        = zeros(3,n);
reference_line = zeros(n,2);

xfit      = double(xrange(ones(41,1),:)./40);
xfit(1,:) = minx;
xfit      = cumsum(xfit);

avgx      = mean(x);
xsettings = avgx(ones(41,1),:);
if (isa(model, 'inline'))
   modname = ['Nonlinear Fit of ' formula(model)];
elseif (isa(model, 'function_handle'))
   modname = 'Nonlinear Fit';
else
   modname = ['Nonlinear Fit of ',model,' Model'];
end
nlin_fig = figure('Units','Normalized','Interruptible','on','Position',[0.05 0.45 0.90 0.5],...
             'Name',modname,'Tag','nlinfig');
set(nlin_fig,'BusyAction','queue','WindowButtonMotionFcn','nlintool(''motion'',0)', ...
             'WindowButtonDownFcn','nlintool(''down'')','WindowButtonUpFcn','nlintool(''up'')', ...
			 'Backingstore','off','WindowButtonMotionFcn','nlintool(''motion'',0)');
yextremes = zeros(n, 2);
addplotdata = isequal(plotdata,'on');
for k = 1:n
   % Create an axis for each input (x) variable
    axisp   = [.18 + (k-1)*.80/n .22 .80/n .68];

    nlin_axes(k) = axes;
    set(nlin_axes(k),'XLim',xlims(k,:),'Box','on','NextPlot','add',...
        'DrawMode','fast','Position',axisp,'GridLineStyle','none');
    if k>1
       set(nlin_axes(k),'Yticklabel',[]);
    end

    % Show data if requested, which implies just one column in x
    if addplotdata
       line(x(:,k),y, 'Linestyle','none', 'Marker','o', 'Parent',nlin_axes(k));
    end

   % Calculate y values for fitted line plot.
   ith_x      = xsettings;
   ith_x(:,k) = xfit(:,k);
   [yfit, deltay] = PredictionsPlusError(ith_x,ud);
 
   % Plot prediction line with confidence intervals
   set(nlin_fig,'CurrentAxes',nlin_axes(k));
   fitline(1:3,k) = plot(ith_x(:,k),yfit,'g-',ith_x(:,k),yfit-deltay,'r--',ith_x(:,k),yfit+deltay,'r--');

   % Calculate data for vertical reference lines
   yextremes(k,:) = get(gca,'YLim');
end  % End of the plotting loop over all the axes.

ymin = min(yextremes(:,1));
ymax = max(yextremes(:,2));

[newy, newdeltay] = PredictionsPlusError(avgx,ud);

for k = 1:n
   set(nlin_axes(k),'YLim',[ymin ymax]);
   xlimits = get(nlin_axes(k),'XLim');
   %   Create Reference Lines
   xvert = repmat(avgx(k), size(yextremes(k,:)));
   set(nlin_fig,'CurrentAxes',nlin_axes(k));
   reference_line(k,1) = plot(xvert,[ymin ymax],'--','Erasemode','xor');
   reference_line(k,2) = plot(xlimits,[newy newy],':','Erasemode','xor');
   set(reference_line(k,1),'ButtonDownFcn','nlintool(''down'')');
end

% Reference lines should not cause the axis limits to extend
set(reference_line(:),'XLimInclude','off','YLimInclude','off');

uihandles = MakeUIcontrols(xstr,nlin_fig,newy,newdeltay,ystr,avgx);
ud.x_field = uihandles.x_field; 
ud.y_field  = uihandles.y_field;

ud.texthandle = [];

ud.xfit = xfit;
ud.nlin_axes = nlin_axes;
ud.xsettings = xsettings;
ud.fitline = fitline;
ud.reference_line = reference_line;
ud.last_axes = zeros(1,n);
ud.wasnan = wasnan;
setconf(ud);           % initialize settings on Bounds menu
set(nlin_fig,'UserData',ud,'HandleVisibility','callback');

if ~isempty(hwarn) && ishandle(hwarn)
   figure(hwarn);
end

% Finished with plot startup function.

case 'motion',
   p = get(nlin_fig,'CurrentPoint');
   k = floor(1+n*(p(1)-0.18)/.80);
   if k < 1 || k > n
      return
   end
   newx(k) = str2double(get(x_field(k),'String'));
   maxx = xrange(k,2);
   minx = xrange(k,1);
     
    if flag == 0,
       % Check for data consistency in each axis
       if n > 1,
          yn = zeros(n,1);
          for idx = 1:n
             y = get(reference_line(idx,2),'Ydata');
             yn(idx) = y(1);
          end
          % If data is inconsistent update all the plots.
          if diff(yn) ~= zeros(n-1,1)
             nlintool('up');
          end
       end
       % Change cursor to plus sign when mouse is on top of vertical line.
        cursorstate = get(nlin_fig,'Pointer');
        cp = get(nlin_axes(k),'CurrentPoint');
        cx = cp(1,1);
        fuzz = 0.02 * (maxx - minx);
        online = cx > newx(k) - fuzz & cx < newx(k) + fuzz;
        if online && strcmp(cursorstate,'arrow'),
            cursorstate = 'crosshair';
        elseif ~online && strcmp(cursorstate,'crosshair'),
            cursorstate = 'arrow';
        end
        set(nlin_fig,'Pointer',cursorstate);
        return
      elseif flag == 1,
        if last_axes(k) == 0
            return;
        end
        cp = get(nlin_axes(k),'CurrentPoint');
        cx=cp(1,1);
        if cx > maxx
            cx = maxx;
        end
        if cx < minx
            cx = minx;
        end

       xrow  = xsettings(1,:);
       xrow(k) = cx;
       [cy, dcy] = PredictionsPlusError(xrow,ud);

        ud.xsettings = xsettings;            
        set(nlin_fig,'Userdata',ud);       
 
        set(x_field(k),'String',num2str(double(cx)));
        set(reference_line(k,1),'XData', repmat(cx,2,1));
        set(reference_line(k,2),'YData',[cy cy]);
      
        set(y_field1,'String',num2str(double(cy)));
        set(y_field2,'String',num2str(double(dcy)));

     end  % End of code for dragging reference lines

case 'down',
       p = get(nlin_fig,'CurrentPoint');
       k = floor(1+n*(p(1)-0.18)/.80);
       if k < 1 || k > n || p(2) > 0.90 || p(2) < 0.22
          return
       end
       ud.last_axes(k) = 1;
       set(nlin_fig,'Pointer','crosshair');
       cp = get(nlin_axes(k),'CurrentPoint');
       cx=cp(1,1);
	   xl = get(nlin_axes(k),'Xlim');
	   maxx = xl(2);
	   minx = xl(1);
       if cx > maxx
          cx = maxx;
       end
       if cx < minx
          cx = minx;
       end

       xrow  = xsettings(1,:);
       xrow(k) = cx;
       xsettings(:,k) = cx(ones(41,1));
       [cy, dcy] = PredictionsPlusError(xrow,ud);

       ud.xsettings = xsettings;            
       set(nlin_fig,'Userdata',ud);       
 
       set(x_field(k),'String',num2str(double(cx)));
       set(reference_line(k,1),'XData',cx*ones(2,1));
       set(reference_line(k,2),'YData',[cy cy]);
      
       set(y_field1,'String',num2str(double(cy)));
       set(y_field2,'String',num2str(double(dcy)));

       set(nlin_fig,'WindowButtonUpFcn','nlintool(''up'')');

case 'up',
   p = get(nlin_fig,'CurrentPoint');
   k = floor(1+n*(p(1)-0.18)/.80);
   lk = find(last_axes == 1);
   if isempty(lk)
      return
   end
   if k < lk
      set(x_field(lk),'String',num2str(double(xrange(lk,1))));
   elseif k > lk
      set(x_field(lk),'String',num2str(double(xrange(lk,2))));
   end
       
    set(nlin_fig,'WindowButtonMotionFcn','nlintool(''motion'',0)');

    updateplot(nlin_fig, ud);

case 'edittext',
   cx    = str2double(get(x_field(flag),'String'));
   
   if isempty(cx)
      set(x_field(flag),'String',num2str(double(xsettings(1,flag))));
      % Create Bad Settings Warning Dialog.
      warndlg('Please type only numbers in the editable text fields.');
      return
   end  
   xl = get(nlin_axes(flag),'Xlim');
   if cx < xl(1) || cx > xl(2)
      % Create Bad Settings Warning Dialog.
      warndlg('This number is outside the range of the data for this variable.');
      set(x_field(flag),'String',num2str(double(xsettings(1,flag))));
      return
   end
   
   last_axes(flag) = 1;
   ud.last_axes = last_axes;
   ud = updateplot(nlin_fig, ud);
   last_axes(flag) = 0;
   ud.last_axes = last_axes;
   set(nlin_fig,'Userdata',ud);       

case 'output',
    bmf = get(nlin_fig,'WindowButtonMotionFcn');
    bdf = get(nlin_fig,'WindowButtonDownFcn');
    set(nlin_fig,'WindowButtonMotionFcn','');
    set(nlin_fig,'WindowButtonDownFcn','');

    labels = {'Parameters', 'Parameter CI', 'Prediction', 'Prediction CI', 'RMSE', 'Residuals'};
    varnames = {'beta', 'betaci', 'ypred', 'ypredci', 'rmse', 'residuals'};

    fullresids = repmat(NaN, size(ud.wasnan));
    fullresids(~ud.wasnan) = ud.residuals;

    % code copied from nlparci
    Rinv = R\eye(size(R));
    diag_info = sum(Rinv.*Rinv,2);
    deltab = sqrt(diag_info) .* ud.crit_t;
    parci = [(beta(:) - deltab) (beta(:) + deltab)];
    yci = [newy-newdeltay newy+newdeltay];
        
    items = {beta, parci, newy, yci, ud.rmse, fullresids};  
    export2wsdlg(labels, varnames, items, 'Export to Workspace');
   
    set(nlin_fig,'WindowButtonMotionFcn',bmf);
    set(nlin_fig,'WindowButtonDownFcn',bdf);

case 'conf',
   if (nargin > 1), conf = flag; end
   switch(conf)
    case 1, ud.simflag = 1;             % simultaneous
    case 2, ud.simflag = 0;             % non-simultaneous
    case 3, ud.obsflag = 0;             % for the mean (fitted line)
    case 4, ud.obsflag = 1;             % for a new observation
    case 5, ud.confflag = ~ud.confflag; % no confidence intervals
   end

   if (ud.simflag)
      ud.crit = ud.crit_f;
   else
      ud.crit = ud.crit_t;
   end

   % Update menu and bounds
   setconf(ud);
   updateplot(nlin_fig, ud);
end


function [y, deltay] = PredictionsPlusError(x,ud)
% Local function for Predicting a response with error bounds.
crit  = ud.crit;
R     = ud.R;
beta  = ud.beta;
model = ud.model;

% Predict Response
y = feval(model,beta,x);
   
% Calculate Error Bounds.
fdiffstep = eps(class(beta)).^(1/3);
J = zeros(length(y),length(beta));
for k = 1:length(beta)
   delta = zeros(size(beta));
   if (beta(k) == 0)
      nb = sqrt(norm(beta));
      delta(k) = fdiffstep * (nb + (nb==0));
   else
      delta(k) = fdiffstep*beta(k);
   end
   yplus = feval(model, beta+delta, x);
   J(:,k) = (yplus - y)/delta(k);
end

E = J / R;
if (ud.obsflag)
   deltay = sqrt(sum(E.*E,2) + 1)*crit;  % predict new observation
else
   deltay = sqrt(sum(E.*E,2))*crit;      % estimate regression function
end

function uihandles = MakeUIcontrols(xstr,nlin_fig,newy,newdeltay,ystr,avgx)
% Local function for Creating uicontrols for nlintool.
fcolor = get(nlin_fig,'Color');
yfieldp = [.01 .45 .10 .04];
uihandles.y_field(1) = uicontrol(nlin_fig,'Style','text','Units','normalized',...
         'Position',yfieldp + [0 .14 0 0],'String',num2str(double(newy)),...
         'ForegroundColor','k','BackgroundColor',fcolor);

uihandles.y_field(2) = uicontrol(nlin_fig,'Style','text','Units','normalized',...
         'Position',yfieldp,'String',num2str(double(newdeltay)),...
         'ForegroundColor','k','BackgroundColor',fcolor);

uicontrol(nlin_fig,'Style','text','Units','normalized',...
         'Position',yfieldp + [0 .07 -0.01 0],'String',' +/-',...
       'ForegroundColor','k','BackgroundColor',fcolor);

uicontrol(nlin_fig,'String', 'Export ...',...
              'Units','pixels','Position',[20 45 100 25],...
              'CallBack','nlintool(''output'')');
uicontrol(nlin_fig,'Style','Pushbutton','Units','pixels',...
               'Position',[20 10 100 25],'Callback','close','String','Close');

uicontrol(nlin_fig,'Style','text','Units','normalized',...
        'Position',yfieldp + [0 0.21 0 0],'BackgroundColor',fcolor,...
        'ForegroundColor','k','String',ystr);

n = length(xstr);
for k = 1:n
   xfieldp = [.18 + (k-0.5)*.80/n - 0.5*min(.5/n,.15) .09 min(.5/n,.15) .07];
   xtextp  = [.18 + (k-0.5)*.80/n - 0.5*min(.5/n,.18) .02 min(.5/n,.18) .05];
   uicontrol(nlin_fig,'Style','text','Units','normalized',...
        'Position',xtextp,'BackgroundColor',fcolor,...
        'ForegroundColor','k','String',xstr{k});

   uihandles.x_field(k)  = uicontrol(nlin_fig,'Style','edit','Units','normalized',...
         'Position',xfieldp,'String',num2str(double(avgx(k))),...
         'BackgroundColor','white','CallBack',['nlintool(''edittext'',',int2str(k),')']);
end

% Create menu for controlling confidence bounds
f = uimenu('Label','&Bounds', 'Position', 4, 'UserData','conf');
uimenu(f,'Label','&Simultaneous', ...
       'Callback','nlintool(''conf'',1)', 'UserData',1);
uimenu(f,'Label','&Non-Simultaneous', ...
       'Callback','nlintool(''conf'',2)', 'UserData',2);
uimenu(f,'Label','&Curve', 'Separator','on', ...
       'Callback','nlintool(''conf'',3)', 'UserData',3);
uimenu(f,'Label','&Observation', ...
       'Callback','nlintool(''conf'',4)', 'UserData',4);
uimenu(f,'Label','N&o Bounds', 'Separator','on', ...
       'Callback','nlintool(''conf'',5)', 'UserData',5);

% -----------------------
function setconf(ud)
%SETCONF Update menus to reflect confidence bound settings

ma = get(findobj(gcf, 'Type','uimenu', 'UserData','conf'), 'Children');
set(ma, 'Checked', 'off');          % uncheck all menu items

% Check item 1 for simultaneous, 2 for non-simultaneous
hh = findobj(ma, 'Type', 'uimenu', 'UserData', 2-ud.simflag);
if (length(hh) == 1), set(hh, 'Checked', 'on'); end

% Check item 3 for curve, 4 for new observation
hh = findobj(ma, 'Type', 'uimenu', 'UserData', 3+ud.obsflag);
if (length(hh) == 1), set(hh, 'Checked', 'on'); end

% Check item 5 if we are omitting confidence bounds.  The remaining 
hh = findobj(ma, 'Type', 'uimenu', 'UserData', 5);
if ud.confflag
    set(ma, 'Enable','on');
else
    set(ma, 'Enable','off');
    set(hh, 'Checked', 'on', 'Enable','on');
end

% -----------------------
function ud = updateplot(nlin_fig, ud)
% Update plot after change in X value or confidence bound setting

x_field       = ud.x_field;
nlin_axes     = ud.nlin_axes;
xsettings     = ud.xsettings;
y_field1      = ud.y_field(1);
y_field2      = ud.y_field(2);
fitline       = ud.fitline;
reference_line= ud.reference_line;
last_axes     = ud.last_axes;  
n             = size(x_field,2);
xfit          = ud.xfit;

% Get current X values, updating from edit box if necessary
xrow  = xsettings(1,:);
lk = find(last_axes == 1);
if (~isempty(lk))
    cx    = str2double(get(x_field(lk),'String'));
    xrow(lk) = cx;
end

[cy, dcy] = PredictionsPlusError(xrow,ud);

% If we need to update reference lines, lk will be non-empty
if (~isempty(lk))
   xsettings(:,lk) = cx(ones(41,1));
   set(reference_line(lk,1),'XData',cx*ones(2,1));
   set(reference_line(lk,2),'YData',[cy cy]);
end

ymax = zeros(n,1);
ymin = zeros(n,1);
ud.xsettings = xsettings;            

for idx = 1:n
   ith_x      = xsettings;
   ith_x(:,idx) = xfit(:,idx);
   [yfit, deltay] = PredictionsPlusError(ith_x,ud);

   if ~ud.confflag
      % No conf bounds wanted, so set their y data to a vector of NaNs
      % so they will not plot but the lines will be around for future use
      deltay = NaN;
   end
   set(nlin_axes(idx),'YlimMode','auto');
   set(fitline(1,idx),'Ydata',yfit);
   set(fitline(2,idx),'Ydata',yfit-deltay);
   set(fitline(3,idx),'Ydata',yfit+deltay);
   ylim = get(nlin_axes(idx),'YLim');
   ymin(idx) = ylim(1);
   ymax(idx) = ylim(2);
end         

ylims = [min(ymin), max(ymax)];

for ix = 1:n      
   ud.last_axes(ix) = 0;
   set(nlin_axes(ix),'Ylim',ylims);
   set(reference_line(ix,1),'Ydata',ylims);
   set(reference_line(ix,2),'YData',[cy cy]);
end
set(nlin_fig,'Userdata',ud);       

set(y_field1,'String',num2str(double(cy)));
set(y_field2,'String',num2str(double(dcy)));