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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

mex IdentifySecPropagateSubfunction.cpp

%%% For a local computer, you don't need to use this CPcompilesubfunction m-file. Instead, open MatLab, change the current directory to CellProfiler/Modules, and type "mex filename.cpp"

%%% For the cluster:
%%% If necessary, change the subfunction name to be compiled in the first line of this file, and re-save this file (a copy of this fresh file must be saved on the cluster machine).
%%% On the cluster cd (change directory) to the CellProfiler/Modules folder. Type in the following in order to run this m-file on a cluster computer:
%%% for mexglx file: bsub -q sq32mp -u xuefang_ma@wi.mit.edu "/nfs/apps/matlab701/bin/matlab -nodisplay -nojvm -c "1700@castlellan.wi.mit.edu"  < ../CPsubfunctions/CPcompilesubfunction.m"
%%% for mexa64 file: bsub -q lq64lp -u xuefang_ma@wi.mit.edu "/nfs/apps/matlab701/bin/matlab -nodisplay -nojvm -c "1700@castlellan.wi.mit.edu"  < ../CPsubfunctions/CPcompilesubfunction.m
%%% 
%%% Notes:
%%% Put your own email address instead of xuefang_ma@wi.mit.edu.
%%% This part specifies the license number for Matlab: -c "1700@castlellan.wi.mit.edu"