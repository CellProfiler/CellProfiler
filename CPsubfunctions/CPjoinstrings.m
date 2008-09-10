function string = CPjoinstrings(varargin)
%CPjoinstrings Build underscore-separated string from parts.
%
%   CPjoinstrings(D1,D2, ... ) builds a string from 
%   D1,D2, etc specified.  This is conceptually equivalent to
%
%      F = [D1 '_' D2 '_' ... '_' DN] 
%
%   Care is taken to handle the cases where the directory
%   parts D1, D2, etc. may contain an empty string, in which case there are
%   not two consecutive underscores output.
%
%   Examples
%   See also FILESEP, PATHSEP, FILEPARTS.

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
% $Revision$

error(nargchk(2, Inf, nargin, 'struct'));

sepchar = '_';
string = varargin{1};

for i=2:nargin,
   part = varargin{i};
   if isempty(string) || isempty(part)
      string = [string part];
   else
      % Handle the three possible cases
      if (string(end)==sepchar) && (part(1)==sepchar),
         string = [string part(2:end)];
      elseif (string(end)==sepchar) || (part(1)==sepchar )
         string = [string part];
      else
         string = [string sepchar part];
      end
   end
end



