function [w, b] = cnnConvolveInit(numFeatures, filterDim)
% CNNINIT - Initializes a convolutional layer with random weights and
% biases.
%
%   [W, B] = CNNCONVOLVEINIT(NUMFEATURES, FILTERDIM)
%
%   INPUTS
%       NUMFEATURES - The number of feature maps extracted by the layer.
%       FILTERDIM - The dimension of the filter to use.
%
%   OUTPUTS
%       W - (filterDim x filterDim x numFeatures) matrix of random weights.
%       B - (numFeatures x 1) vector of random biases.
w = rand(filterDim, filterDim, numFeatures);
b = rand(numFeatures, 1);
end
