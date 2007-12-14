function CPupdatefigurecycle(SetBeingAnalyzed,FigureNumber);

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
% $Revision: 4931 $

OldText = get(FigureNumber,'name');
NumberSignIndex = find(OldText=='#');
OldTextUpToNumberSign = OldText(1:NumberSignIndex(1));
NewNum = SetBeingAnalyzed;
set(FigureNumber,'name',[OldTextUpToNumberSign,num2str(NewNum)]);