addpath('third-party/mnistHelper');
addpath('matconvnet/matlab');
run('matconvnet/matlab/vl_setupnn.m');
dataDir = 'mnist';

% ########################
% DATA LOADING
% ########################

trainImages = loadMNISTImages(fullfile(dataDir, 'train-images-idx3-ubyte'));
trainLabels = loadMNISTLabels(fullfile(dataDir, 'train-labels-idx1-ubyte'));
testImages = loadMNISTImages(fullfile(dataDir, 't10k-images-idx3-ubyte'));
testLabels = loadMNISTLabels(fullfile(dataDir, 't10k-labels-idx1-ubyte'));

processLabels = @(l) l' + 1;
processImages = @(im) single(reshape(im, size(im, 1), size(im, 2), 1, []));

trainImages = processImages(trainImages);
trainLabels = processLabels(trainLabels);
testImages = processImages(testImages);
testLabels = processLabels(testLabels);

% ########################
% TRAINING
% ########################

batchSize = 256;
% net = cnnMNISTSGD(trainImages, trainLabels, @getBatch, 'batchSize', batchSize);

% ########################
% PREDICTION
% ########################

predictions = zeros(size(testLabels));
numBatches = ceil(size(testLabels, 2) / batchSize);
net.layers(end-1:end) = [];
pidx = 1;

for b = 1:numBatches
    fprintf('Predicting batch %d/%d\n', b, numBatches);
    [imb, ~] = getBatch(testImages, testLabels, batchSize, b);
    res = vl_simplenn(net, imb);
    
    outputs = squeeze(res(end).x);
    numOutputs = size(outputs, 2);
    for i = 1:numOutputs
        [~, maxIdx] = max(outputs(:, i));
        predictions(pidx) = maxIdx;
        pidx = pidx + 1;
    end
end
