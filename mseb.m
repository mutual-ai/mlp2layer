function [ E ] = mseb(y,yr)
% Mean Square Error Batch Mode
% y - training set outputs
% yr - training set reference outputs 

[~,n] = size(y);
 E = ( (yr-y).^2 );
 E = E';
 E = sum(sum(E)/n)/n;
end
