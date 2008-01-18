function CPupdatefigurecycle(SetBeingAnalyzed,FigureNumber);

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
% $Revision: 4931 $

OldText = get(FigureNumber,'name');
NumberSignIndex = find(OldText=='#');
OldTextUpToNumberSign = OldText(1:NumberSignIndex(1));
NewNum = SetBeingAnalyzed;
set(FigureNumber,'name',[OldTextUpToNumberSign,num2str(NewNum)]);