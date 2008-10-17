path('/Users/yoavfreund/projects/MCF7/matlab',path)
imagefilename= '/Users/yoavfreund/projects/MCF7/images/MCF7_JDO76_S22_T023'
sz= 25;    				% size of the box
global as; as= 10; 			% angle step (degrees)

step_sz = 5;				% position step (pixels)

% [I,D,mask] = getRotatedImagesAndMask(imagefilename);

scores = calculateScores(I,D,mask,sz,step_sz,as);
