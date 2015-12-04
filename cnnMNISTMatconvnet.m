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

% batchSize = 32;

fprintf('Starting Test\n');
fprintf('%20s %20s %20s %20s %20s\n','Network','Batch Size','Epochs','Learning Rate','Accuracy');
fprintf('------------------------------------------------------------------------------------------------------------------------------\n');

for netMatFile=dir('testnet/*.mat')'
    netIn = load(strcat('testnet/',netMatFile.name),'net');
    netIn = netIn.net;
    for batchSize=[32,64,128,256,512,1028]
        for epochs=[10,20,30,40,50]
            for learningRate=[0.0001,0.001,0.01,0.1]
                for momentum=[[0.5,repmat(0.9,epochs,1)'];...
                        [0.5,0.9:(0.95-0.9)/(epochs-1):0.95];...
                        [0.5,0.9:(0.92-0.9)/(epochs-1):0.92];...
                        repmat(0.9,epochs+1,1)']';
                    net = cnnMNISTSGD(trainImages, trainLabels, @getBatch, 'batchSize', batchSize,...
                        'momentum', momentum, 'numEpochs',epochs, 'net', netIn, 'learningRate',learningRate);
                    
                    % ########################
                    % PREDICTION
                    % ########################
                    
                    result.netIn = netIn;
                    result.net = net;
                    result.batchSize = batchSize;
                    result.epochs = epochs;
                    result.learningRate = learningRate;
                    result.momentum = momentum;
                    
                    predictions = predictWithRandomCrop(net, testImages, testLabels, 10, cx, cy, 4);
                    acc = sum(predictions == testLabels) / numel(testLabels);
                    
                    
                    fprintf('%20s %20d %20d %20d %20s %20.2f%%\n',netIn,batchSize,epochs,learningRate, acc * 100);
                    fprintf('Momentum: '); disp(momentum);
                    fprintf('\n\n');
                    
                    result.acc = acc;
                    
                    save(strcat('output/','result_',length(dir('output/*.mat')) + 1,'.mat'),'result');
                    
                end
            end
        end
    end
end

