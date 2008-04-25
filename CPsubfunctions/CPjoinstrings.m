% CPJOINSTRINGS Convert arguments to strings and join together with underscore.
%    The function takes a variable number of argument.  Each argument can be
%    a string (i.e., a character array) or a number.
%
%    Examples:
%       s = CPjoinstrings() returns the empty string.
%       s = CPjoinstrings('foo') returns 'foo'.
%       s = CPjoinstrings('foo', 42) returns 'foo_42'.
%       s = CPjoinstrings('foo', 23, 'bar') returns 'foo_23_bar'.
function string = CPjoinstrings(varargin)
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision: 5139 $
string = '';
for i = 1:nargin
  if i > 1
    string = [string, '_'];
  end
  arg = varargin{i};
  if ischar(arg)
    fragment = arg;
  else
    fragment = num2str(arg);
  end
  string = [string, fragment];
end
