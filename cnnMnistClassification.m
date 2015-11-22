addpath('third-party/mnistHelper');

numSamples = 100;

train_images = loadMNISTImages('mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
train_images = train_images(:, :, 1:numSamples);
train_labels = train_labels(1:numSamples);
n = size(train_labels, 1);

% ==============
% CONFIGURATION
% ==============

nEpochs = 1;
activationFn = 'logsig';

% INPUT: 32x32 image
% Layer 1: Convolution, 6 features, 5x5 neighbourhood, 28x28 output
l1Features = 6;
l1FilterDim = 5;

% Layer 2: Subsampling, 6 features, 2x2 neighbourhood, 14x14 output
l2PoolDim = 2;

% Layer 3: Convolution, 16 features, 5x5 neighbourhood, 10x10 output
l3Features = 16;
l3FilterDim = 5;

% Layer 4: Subsampling, 16 features, 2x2 neighbourhood, 5x5 output
l4PoolDim = 2;

% Layer 5: Convolution, 120 features, 5x5 neighbourhood, 1x1 output
l5Features = 120;
l5FilterDim = 5;

% Layer 6: Fully connected, 84 neurons.
l6Neurons = 84;

% OUTPUT: 10 neurons

% ==============
% PREPROCESSING
% ==============

% Pad images to 32x32 for the reason described in the LeCun paper (1998):
%
% "The input is a 32x32 pixel image. This is significantly larger than the 
% largest character in the database (at most 20x20 pixels centered in a 
% 28x28 field). The reason is that it is desirable that potential 
% distinctive features such as stroke end-points or corner can appear in 
% the center of the receptive field of the highest-level feature
% detectors."
train_images = padarray(train_images, [2 2 0]);

% ==============
% TRAINING
% ==============

% Mapping of features from the subsampling layer 2 and the
% convolutional layer 3. Each column indicates which feature
% maps in layer two are connected by units in a particular
% feature map of layer 3. From the LeCun paper.
l23Map = [1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1; ...
          1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1; ...
          1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1; ...
          0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1; ...
          0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1; ...
          0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1];

% Initialize initial weights and biases.
[w1, b1] = cnnConvolveInit(l1Features, l1FilterDim);
[w3, b3] = cnnConvolveInit(l3Features, l3FilterDim);
[w5, b5] = cnnConvolveInit(l5Features, l5FilterDim);
    
for i = 1:nEpochs
    % ==============
    % FEED FORWARD
    % ==============
    
    l1Features = cnnConvolve(train_images, w1, b1, activationFn);
    l2Features = cnnPool(l2PoolDim, l1Features);
    l3Features = cnnConvolve(l2Features, w3, b3, activationFn);
    l4Features = cnnPool(l4PoolDim, l3Features);
    l5Features = cnnConvolve(l4Features, w5, b5, activationFn);
end
