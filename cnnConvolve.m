function features = cnnConvolve(images, w, b, activationFn)
% CNNCONVOLVE - Returns features obtained by convolving filters represented
% by learned weights & bias with images.
%
%   FEATURES = CNNCONVOLVE(IMAGES, W, B, ACTIVATIONFN)
%
%   INPUTS
%       IMAGES - a (imageDim x imageDim x numImages) matrix of square, single
%       channel images. imageDim is the dimension (width and height) of the
%       square image, and numImages is the number of images.
%       W - a (filterDim x filterDim x numFilters) matrix of weights.
%       filterDim is the dimension of the filter to convolve with, and
%       numFilters is the number of filters.
%       B - a (numFilters, 1) vector of bias values.
%
%   OUTPUTS
%       FEATURES - a (convDim x convDim x filterNum x imageNum) matrix of
%       features, where convDim = imageDim - filterDim + 1.
%   
%   REFERENCES
%       * http://ufldl.stanford.edu/tutorial/supervised/ExerciseConvolutionAndPooling/

filterDim = size(w, 1);
numFilters = size(w, 3);
imageDim = size(images, 1);

assert(filterDim == size(w, 2), 'First two dimensions of weight matrix must be equal');
assert(isequal([numFilters 1], size(b)), 'b must be a numFiltersx1 vector, where numFilters is the third dimension of the weight matrix');
assert(imageDim == size(images, 2), 'Images must be square');
assert(ndims(images) == 3, 'Images can only have a single channel');

% Dimension to convolve inside such that the entire patch is contained
% within the image instead of extending beyond the edges of the image.
convDim = imageDim - filterDim + 1;
numImages = size(images, 3);
f = getActivationFn(activationFn);
features = zeros(convDim, convDim, numImages, numFeatures);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        feature = w(:, :, filterNum);
        feature = rot90(feature, 2);
        image = images(:, :, imageNum);
        convolvedImage = conv2(image, feature, 'valid');
        features(:, :, filterNum, imageNum) = f(convolvedImage + b(filterNum));
    end
end
end
