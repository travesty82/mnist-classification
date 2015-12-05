addpath('third-party/mnistHelper');
addpath('matconvnet/matlab');
run('matconvnet/matlab/vl_setupnn.m');
dataDir = 'mnist';

rng('default');
rng(0);

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
trainImages = single(trainImages);
croppedTrainImages = single(cropImageBatchRandom(trainImages, cx, cy));

% ########################
% TRAINING
% ########################

useBnorm = [0, 1];
useCropping = [0, 1];
learningRate = [0.001, 0.01, 0.1];
netID = [1, 2, 3];
method = [1, 2];

results = zeros(1, 6);
for nid = netID
    if nid == 3
        bs = 100;
    else
        bs = 96;
    end
    for m = method
        for lr = learningRate
            for bn = useBnorm
                for cr = useCropping
                    rng('default');
                    rng(0);
                    if cr == 1
                        images = croppedTrainImages;
                    else
                        images = trainImages;
                    end
                    if m == 1
                        net = cnnMNISTSGD(images, trainLabels, @getBatch, ...
                            'batchSize', bs, 'learningRate', lr, ...
                            'useCropping', cr, 'useBnorm', bn, 'netID', nid);
                    else
                        net = cnnMNISTAdam(images, trainLabels, @getBatch, ...
                            'batchSize', bs, 'learningRate', lr, ...
                            'useCropping', cr, 'useBnorm', bn, 'netID', nid);
                    end
                    if cr
                        predictions = predictWithRandomCrop(net, testImages, testLabels, 10, cx, cy, 4);
                    else
                        net.layers{end}.class = testLabels;
                        res = vl_simplenn(net, testImages, [], [], 'disableDropout', true);
                        predictions = squeeze(res(end - 1).x);
                        [~, predictions] = sort(predictions, 1, 'descend');
                        predictions = reshape(predictions(1, :), 1, []);
                    end
                    acc = sum(predictions == testLabels) / numel(testLabels);
                    
                    fprintf('method=%d,lr=%d,bn=%d,cr=%d,acc=%.4f\n', m, lr, bn, cr, acc);
                    results(end + 1, :) = [nid m lr bn cr acc];
                end
            end
        end
    end
end
