function [I,D] = rotateAll(ImageList, DerivList)
% read a set of images, all corrsponding to the same Field of View, and
% generate rotated versions of them. Rotations are counter-clockwise with
% steps of size "as".
%    
% ImageList - list of grey level images.
% DerivList - list of derivative images (magnitude + direction)
% I cell array pointing to the arrays with the rotated grey-level images
% D structure array pointing to the array-pairs with rotated derive
% images
    global as;                          % angle step

    I={}; 
    D{1,1}.angle=[];
    D{1,1}.mag=[];
    
    if(length(ImageList)>0)
        for i=1:length(ImageList)
            A=imread(ImageList{i});
            I{i,1}=A;
            for ai=1:(90/as)-1
                I{i,ai+1} = imrotate(A,ai*as,'bilinear');
            end
        end
    end
    
    if(length(DerivList)>0)
        for i=1:length(DerivList)
            A=imread(DerivList{i});
            HSV=rgb2hsv(A);
            H=HSV(:,:,1);
            S=HSV(:,:,2);
            D{i,1}.angle = H;
            D{i,1}.mag = S;
            
            for ai=1:(90/as)-1
                D{i,ai+1}.angle = imrotate(mod(H-ai*(as/360.),1),ai*as);
                D{i,ai+1}.mag = imrotate(S,ai*as,'bilinear');
%                 clear HSV;
%                 HSV(:,:,1)=D{i,ai+1}.angle;
%                 HSV(:,:,2)=D{i,ai+1}.mag;
%                 HSV(:,:,3)=1;
%                 imtool(hsv2rgb(HSV));
            end
        end
    end
    
    
