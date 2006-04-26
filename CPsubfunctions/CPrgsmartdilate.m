function rgOut = RgSmartDilate(rgIn,n)
% Accepts labelled regions and dilates them with radius 1 diamond, disallowing region overlap. 
% rgIn is a labelled region image, n is number of iterations.
%
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
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

global rg4trans

intwarning('off');

rgOut=rgIn;
for i=1:n
    %generate array of neighboring values 
    %Note unstylish kluge at start of RunBatch.  For a standalone
    %function (costs 1 second) use
    %   rg4trans=uint16(zeros([size(rgOut) 4]));
    rg4trans=uint16(zeros([size(rgOut) 4]));
    rg4trans(:,:,:)=0;
    %Shift up
    rg4trans(1:end-1,1:end  ,1)=rgOut(2:end  ,1:end);
    %Shift down
    rg4trans(2:end  ,1:end  ,2)=rgOut(1:end-1,1:end);
    %Shift left
    rg4trans(1:end  ,1:end-1,3)=rgOut(1:end  ,2:end);
    %Shift right
    rg4trans(1:end  ,2:end  ,4)=rgOut(1:end  ,1:end-1);
    
    
    %find maximal neigbor value
    rgMax=max(rg4trans,[],3);
    %find minimal nonzero neighbor value
    rg4trans(rg4trans==0)=Inf;
    rgMin=min(rg4trans,[],3);

    %if only one nonzero neighbor value, return
    rgMax(rgMax~=rgMin)=0;
    
    %if original value was nonzero, return
    rgMax(rgOut>0)=0;
    rgOut=uint16(double(rgOut)+double(rgMax));
    %disp(['Dilated Smartly. Time: ' num2str(toc)])
end
