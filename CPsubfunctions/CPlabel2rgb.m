function im=CPlabel2rgb(handles,im)

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

%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".

if sum(sum(im)) >= 1
    cmap = eval([handles.Preferences.LabelColorMap '(max(2,max(im(:))))']);
    im = label2rgb(im, cmap, 'k', 'shuffle');
else
    im=im;
end