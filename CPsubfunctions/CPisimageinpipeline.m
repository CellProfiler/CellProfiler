function ispresent = CPisimageinpipeline(handles, fieldname)
% Check if images exist in handles.Pipeline structure.

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

ispresent = isfield(handles.Pipeline,fieldname);
