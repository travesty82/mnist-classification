function features = cnnPool(poolDim, convolvedFeatures)
% CNNPOOL - Pools convolved features using mean pooling.
%
% FEATURES = CNNPOOL(POOLDIM, CONVOLVEDFEATURES)
%
%   INPUTS
%       POOLDIM - The dimension of the pool (i.e. the dimension of the
%       arithmetic mean filter to convolve with)
%       CONVOLVEDFEATURES - Convolved features obtained using a call to the
%       CNNCONVOLVE function, with dimensions (convDim x convDim x
%       filterNum x imageNum)
%
%   OUTPUTS
%       FEATURES - Pooled features with dimensions (convDim / poolDim x 
%       convDim / poolDim x filterNum x imageNum)
%
%   REFERENCES
%       * http://ufldl.stanford.edu/tutorial/supervised/Pooling/
%       * http://ufldl.stanford.edu/tutorial/supervised/ExerciseConvolutionAndPooling/
%       
numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

filterDim = convolvedDim / poolDim;
features = zeros(filterDim, filterDim, numFilters, numImages);
meanFilter = ones(filterDim) / (filterDim * filterDim);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        feature = convolvedFeatures(:, :, filterNum, imageNum);
        averagedFeature = conv2(feature, meanFilter, 'valid');
        features(:, :, filterNum, imageNum) = averagedFeature;
    end
end
end
