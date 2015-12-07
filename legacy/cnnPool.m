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
%       numFilters x numImages)
%
%   OUTPUTS
%       FEATURES - Pooled features with dimensions (convDim / poolDim x 
%       convDim / poolDim x numFilters x numImages)
%
%   REFERENCES
%       * http://ufldl.stanford.edu/tutorial/supervised/Pooling/
%       * http://ufldl.stanford.edu/tutorial/supervised/ExerciseConvolutionAndPooling/
%       
numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

featureDim = convolvedDim / poolDim;
features = zeros(featureDim, featureDim, numFilters, numImages);
meanFilter = ones(poolDim) / (poolDim * poolDim);
% From MATLAB conv2() docs, for 'valid' option:
% size(C) = max([ma-max(0,mb-1),na-max(0,nb-1)],0)
featureIdx = 1:poolDim:(convolvedDim - max(0, poolDim - 1));

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        feature = convolvedFeatures(:, :, filterNum, imageNum);
        meanFeature = conv2(feature, meanFilter, 'valid');
        features(:, :, filterNum, imageNum) = meanFeature(featureIdx, featureIdx);
    end
end
end
