function [FigureIncrement, results] = CPec50(conc,responses,LogOrLinear,PartialFigureName,ModuleName,ConcName,FigureIncrement)
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

% Copyright 2004 Carlos Evangelista
% send comments to CCEvangelista@aol.com
% Version 1.0    01/07/2004

%%% If we are using a log-domain set of doses, we have a better chance of
%%% fitting a sigmoid to the curve if the concentrations are
%%% log-transformed.
if strcmpi(LogOrLinear,'Yes')
    conc = log(conc);
end

n=size(responses,2);
results=zeros(n,4);
for i=1:n
    response=responses(:,i);
    initial_params=calc_init_params(conc,response);
    % OPTIONS = statset('display','iter');
    %%% Turns off MATLAB-level warnings but saves the previous warning state.
    PreviousWarningState = warning('off', 'all');
    try
        [coeffs,r,J]=nlinfit(conc,response,'CPsigmoid',initial_params);
    catch
        tes=eps;
    end
    if ~strcmpi(PartialFigureName,'Do not save')
        %%% This produces the figure with the interactive graph.
        if strcmpi(LogOrLinear,'Yes')
            XaxisLabel = ['log(',ConcName,')'];
        else XaxisLabel = [ConcName,''];
        end
        YaxisLabel = ['Feature #',num2str(FigureIncrement)];
        FigureHandle = CPnlintool(conc,response,'CPsigmoid',initial_params,.05,XaxisLabel,YaxisLabel);
        try saveas(FigureHandle,[PartialFigureName,num2str(FigureIncrement),'.fig'],'fig');
            try close(FigureHandle)
            end
        catch
            errordlg(['Image processing was NOT canceled in the ', ModuleName, ' module, but the figure could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
        FigureIncrement = FigureIncrement + 1;
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