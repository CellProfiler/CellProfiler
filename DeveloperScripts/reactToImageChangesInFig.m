function reactToImageChangesInFig(varargin)
%reactToImageChangesInFig Set up reaction to image changes in a figure.
%   reactToImageChangesInFig(HIMAGE,DELETE_FCN) calls DELETE_FCN if the target
%   image HIMAGE changes in its figure.  DELETE_FCN is a function handle
%   specified by the client that would cause the client or some part of the
%   client to delete itself. HIMAGE can be array of handles to graphics image
%   objects.
%
%   reactToImageChangesInFig(HIMAGE,DELETE_FCN,RECREATE_FCN) calls
%   the DELETE_FCN if the target image HIMAGE is deleted from the figure and
%   calls RECREATE_FCN if a new image is added to that figure.  For example,
%   if a user calls imshow twice on the same figure, the target image HIMAGE
%   is deleted and a new image is added. RECREATE_FCN is a function handle
%   specified by the client that would cause the client or some part of the
%   client to recreate itself.
%
%   Notes
%   -----
%   User has the responsibility to use valid function handles.
%   HIMAGE must be an array of image handles that are in the same figure.

%   See also IMPIXELINFOPANEL,IMPIXELINFOVAL.

%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision: 1.1.8.1 $  $Date: 2004/08/10 01:50:34 $

[hImage,deleteFcn,recreateFcn] = parseInputs(varargin{:});


hImageParent = get(hImage,'Parent');
hFig = ancestor(hImage,'Figure');

% an array of images must belong to the same figure.
if iscell(hFig) 
  if any(diff([hFig{:}]))
    eid = sprintf('Images:%s:invalidImageArray',mfilename);
    msg = 'HIMAGE must belong to the same figure.';
    error(eid,'%s',msg);
  else
    hFig = hFig{1};
  end
end

% must detect deletion of himage for either action.
if iscell(hImageParent)
  handleParent = handle([hImageParent{:}]);
else
  handleParent = handle(hImageParent);
end

childRemovedListener = handle.listener(handleParent,'ObjectChildRemoved', ...
                                       {@childRemoved,deleteFcn});
storeListener(handleParent,childRemovedListener);

if ~isempty(recreateFcn)
  childAddedListener = handle.listener(hFig,'ObjectChildAdded', ... 
                                       @childAddedToFig);
  storeListener(hFig,childAddedListener);
end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   function childAddedToFig(obj,eventData)
   % This function is called if a
   % child is added to hFig
      
      if isa(eventData.Child,'axes')
        hAxes = eventData.Child;
        childAddedListener = handle.listener(hAxes,'ObjectChildAdded', ...
                                             @childAddedToAxes);
        storeListener(hAxes,childAddedListener);
      end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function childAddedToAxes(obj,eventData)
       % This function is called if a
       % child is added to hFig
      
          if isa(eventData.Child,'image')
            setPropertyListenerOnImage;
          
          elseif isa(eventData.Child,'hggroup')
            hHGGroup = eventData.Child;
            childAddedListener = handle.listener(hHGGroup,'ObjectChildAdded', ...
                                             @childAddedToHGgroup);
            storeListener(hHGGroup,childAddedListener);
          end
       
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          function childAddedToHGgroup(obj,eventData)
      
             if isa(eventData.Child,'image')
               setPropertyListenerOnImage;
             end
          end
      
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          function setPropertyListenerOnImage
             
             hIm = eventData.Child;
             prop = hIm.findprop('CData');
             propListener = handle.listener(hIm, prop, ...
                                            'PropertyPostSet',@propDone);
             storeListener(hIm,propListener);


             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             function propDone(obj,eventData)
                
                recreateFcn();
             
             end %propDone
          end %setPropertyListenerOnImage
       end %childAddedToAxes           
   end %childAdded     


end %main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [him,deletefcn,recreatefcn] = parseInputs(varargin)

recreatefcn = '';

iptchecknargin(2,3,nargin,mfilename);

him = varargin{1};
checkImageHandleArray(him,mfilename);

deletefcn = varargin{2};

if nargin == 3
  recreatefcn = varargin{3};
end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function storeListener(source,listener)

for k = 1 : numel(source)
  listenerArray = getappdata(source(k),'imageChangeListeners');
  listenerArray = [listenerArray listener];
  setappdata(source(k),'imageChangeListeners',listenerArray);
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function childRemoved(obj,eventData,deleteFcn)
% This function is called if any child in hImageParent in hFig is
% deleted (e.g., you delete a handle to an image, you call imshow again
% using the same figure, etc.)
  
if isa(eventData.Child,'image')
  deleteFcn();
end
   
end
