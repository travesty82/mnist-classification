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

% ########################
% PREPROCESSING
% ########################

processLabels = @(l) l' + 1;
processImages = @(im) reshape(im, size(im, 1), size(im, 2), 1, []);

trainImages = processImages(trainImages);
trainLabels = processLabels(trainLabels);
testImages = processImages(testImages);
testLabels = processLabels(testLabels);

cx = 24; cy = 24;
testImages = single(testImages);
trainImages = single(cropImageBatchRandom(trainImages, cx, cy));

% ########################
% TRAINING
% ########################

batchSize = 32;
%net = cnnMNISTSGD(trainImages, trainLabels, @getBatch, 'batchSize', batchSize);

% ########################
% PREDICTION
% ########################

predictions = predictWithRandomCrop(net, testImages, testLabels, 10, cx, cy, 4);
acc = sum(predictions == testLabels) / numel(testLabels);
fprintf('Accuracy: %.2f%%\n', acc * 100);
