function rgNegCurve = LoGsegmenter(imDAPIin)

NeighborhoodSize = [9 9];
Sigma = 1.8;
MinArea = 3;
wiendim=[5 5];
Threshold =  -.001

%%% Creates the Laplacian of a Gaussian filter.
rgLoG=fspecial('log',NeighborhoodSize,Sigma);
%%% Filters the image.
imLoGout=imfilter(double(imDAPIin),rgLoG);
    figure, imagesc(imLoGout), colormap(gray), title('imLoGout')
%%% Removes noise using the weiner filter.
imLoGoutW=wiener2(imLoGout,wiendim);
    figure, imagesc(imLoGoutW), colormap(gray), title('imLoGoutW')

%%%%%%%%%%%%%%
rgNegCurve = imLoGoutW < Threshold;
class(rgNegCurve)
min(min(rgNegCurve))
max(max(rgNegCurve))

%set outsides
rgNegCurve([1 end],1:end)=1;
rgNegCurve(1:end,[1 end])=1;

%disp(['Generated LoG regions. Time: ' num2str(toc)])

%Throw out noise, label regions
rgArOpen=bwareaopen(rgNegCurve,MinArea,4); 
rgLabelled=uint16(bwlabel(rgArOpen,4));
if max(rgLabelled(:))==1
    error('Error: No DAPI regions generated');
end

%Get rid of region around outsides (upper-left region gets value 1)
rgLabelled(rgLabelled==1)=0; 
rgLabelled(rgLabelled==0)=1;
rgLabelled=uint16(double(rgLabelled)-1);
%disp(['Generated labelled, size-excluded regions. Time: ' num2str(toc)])

%(Smart)closing
rgDilated=RgSmartDilate(rgLabelled,50); %%% IMPORTANT VARIABLE
rgFill=imfill(rgDilated,'holes');

%%%%SE=strel('diamond',1);
%%%%rgErode=imerode(rgFill,SE);
%%%%rgOut=rgErode;

rgOut=rgFill;

%%%%%%%%%%%%

%%% Creates label matrix image.
    rgLabelled2=uint16(bwlabel(imLoGoutW,4));
    figure, imagesc(rgLabelled2), colormap(gray), title('rgLabelled2')

    %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
% What follows is from the file: ImDAPI2Rg.m that Zach also gave me.
% It's probably very similar to the above function.
    
    function [rgOut, varargout] = ImDAPI2Rg(imDAPIin, LoGDim, LoGHW, MinArea)
%Calculate LoG, adaptive noise filter, select negative values
%rgOut (rgNegCurve, imLabelBounds) =ImDAPI2Rg(imDAPIin, LoGDim, LoGHW, MinArea)


wiendim=[5 5];

%tic
%disp('Calculating regions from DAPI')
rgLoG=fspecial('log',LoGDim,LoGHW);
imLoGout=imfilter(double(imDAPIin),rgLoG);
imLoGoutW=wiener2(imLoGout,wiendim);
rgNegCurve=imLoGoutW<-1;

%set outsides
rgNegCurve([1 end],1:end)=1;
rgNegCurve(1:end,[1 end])=1;

%disp(['Generated LoG regions. Time: ' num2str(toc)])

%Throw out noise, label regions
rgArOpen=bwareaopen(rgNegCurve,MinArea,4); 
rgLabelled=uint16(bwlabel(rgArOpen,4));
if max(rgLabelled(:))==1
    error('Error: No DAPI regions generated');
end

%Get rid of region around outsides (upper-left region gets value 1)
rgLabelled(rgLabelled==1)=0; 
rgLabelled(rgLabelled==0)=1;
rgLabelled=uint16(double(rgLabelled)-1);
%disp(['Generated labelled, size-excluded regions. Time: ' num2str(toc)])

%(Smart)closing
rgDilated=RgSmartDilate(rgLabelled,2);
rgFill=imfill(rgDilated,'holes');

%%%%SE=strel('diamond',1);
%%%%rgErode=imerode(rgFill,SE);
%%%%rgOut=rgErode;

rgOut=rgFill;

%If asked for, return ~rgDilated (proxy for rgPosCurve)
if nargout==2
    varargout{1}=~rgDilated;
end
        
%If asked for, make picture with boundaries drawn
if nargout==3
    bwBound=zeros(size(rgOut));
    bwRgnTmp=zeros(size(rgOut));
    for i=1:max(rgOut(:))
        bwRgnTmp(bwRgnTmp==1)=0;
        bwRgnTmp(rgOut==i)=1;
        bwBoundtmp=bwperim(bwRgnTmp);
        bwBound=logical(bwBound+bwBoundtmp);

        varargout{2}=imDAPIin;
        varargout{2}(bwBound)=double(max(imDAPIin(:)))+50;
    end
end
