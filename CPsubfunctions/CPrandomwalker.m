function [mask,probabilities] = random_walker(img,seeds,labels,beta)
%
% Leo Grady's random walker segmentation algorithm.
% From http://cns.bu.edu/~lgrady/
% Uses the graph analysis toolbox (in
% CPsubfunctions/graphAnalysisToolbox-1.0).
%
% This function uses the random walker segmentation algorithm to produce a
% segmentation given a 2D image, input seeds and seed labels.
%
%Inputs: img - The image to be segmented
%        seeds - The input seed locations (given as image indices, i.e., 
%           as produced by sub2ind)
%        labels - Integer object labels for each seed.  The labels 
%           vector should be the same size as the seeds vector.
%        beta - Optional weighting parameter (Default beta = 90)
%
%Output: mask - A labeling of each pixel with values 1-K, indicating the
%           object membership of each pixel
%        probabilities - Pixel (i,j) belongs to label 'k' with probability
%           equal to probabilities(i,j,k)
%
%
%10/31/05 - Leo Grady
%Based on the paper:
%Leo Grady and Gareth Funka-Lea, "Multi-Label Image Segmentation for
%   Medical Applications Based on Graph-Theoretic Electrical Potentials", 
%   in Proceedings of the 8th ECCV04, Workshop on Computer Vision Approaches 
%   to Medical Image Analysis and Mathematical Methods in Biomedical Image 
%   Analysis, p. 230-245, May 15th, 2004, Prague, Czech Republic, 
%   Springer-Verlag
%Available at: http://cns.bu.edu/~lgrady/grady2004multilabel.pdf
%
%Note: Requires installation of the Graph Analysis Toolbox available at:
%http://eslab.bu.edu/software/graphanalysis/

%Read inputs
if nargin < 4
    beta = 90;
end

%Find image size
img=im2double(img);
[X Y Z]=size(img);

%Error catches
exitFlag=0;
if((Z~=1) && (Z~=3)) %Check number of image channels
    disp('ERROR: Image must have one (grayscale) or three (color) channels.')
    exitFlag=1;
end 
if(sum(isnan(img(:))) || sum(isinf(img(:)))) %Check for NaN/Inf image values
    disp('ERROR: Image contains NaN or Inf values - Do not know how to handle.')
    exitFlag=1;
end
%Check seed locations argument
if(sum(seeds<1) || sum(seeds>size(img,1)*size(img,2)) || (sum(isnan(seeds)))) 
    disp('ERROR: All seed locations must be within image.')
    disp('The location is the index of the seed, as if the image is a matrix.')
    disp('i.e., 1 <= seeds <= size(img,1)*size(img,2)')
    exitFlag=1;
end
if(sum(diff(sort(seeds))==0)) %Check for duplicate seeds
    disp('ERROR: Duplicate seeds detected.')
    disp('Include only one entry per seed in the "seeds" and "labels" inputs.')
    exitFlag=1;
end
TolInt=0.01*sqrt(eps);
if(length(labels) - sum(abs(labels-round(labels)) < TolInt)) %Check seed labels argument
    disp('ERROR: Labels must be integer valued.');
    exitFlag=1;
end
if(length(beta)~=1) %Check beta argument
    disp('ERROR: The "beta" argument should contain only one value.');
    exitFlag=1;
end
if(exitFlag)
    disp('Exiting...')
    [mask,probabilities]=deal([]);
    return
end

%Build graph
[points edges]=lattice(X,Y);

%Generate weights and Laplacian matrix
if(Z > 1) %Color images
    tmp=img(:,:,1);
    imgVals=tmp(:);
    tmp=img(:,:,2);
    imgVals(:,2)=tmp(:);
    tmp=img(:,:,3);
    imgVals(:,3)=tmp(:);
else
    imgVals=img(:);
end
weights=makeweights(edges,imgVals,beta);
L=laplacian(edges,weights);
%L=laplacian(edges,weights,length(points),1);

%Determine which label values have been used
label_adjust=min(labels); labels=labels-label_adjust+1; %Adjust labels to be > 0
labels_record(labels)=1;
labels_present=find(labels_record);
number_labels=length(labels_present);

%Set up Dirichlet problem
boundary=zeros(length(seeds),number_labels);
for k=1:number_labels
    boundary(:,k)=(labels(:)==labels_present(k));
end

%Solve for random walker probabilities by solving combinatorial Dirichlet
%problem
probabilities=dirichletboundary(L,seeds(:),boundary);

%Generate mask
[dummy mask]=max(probabilities,[],2);
mask=labels_present(mask)+label_adjust-1; %Assign original labels to mask
mask=reshape(mask,[X Y]);
probabilities=reshape(probabilities,[X Y number_labels]);