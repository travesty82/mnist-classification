function net = cnnMNISTInitMatconvnet(opts)
% CNNMNISTINITMATCONVNET - Initializes a CNN based on the CNN included in the
% MatConvNet examples for MNIST training
% (matconvnet/examples/cnn_mnist_init.m)
%
% INPUTS
%   opts - Configuration options
%       opts.useCropping - Whether data was preprocessed by randomly
%       cropping windows (data augmentation)
%       opts.useBnorm - Whether to insert batch normalization layers after
%       convolution layers
% OUTPUTS
%   net - Initialized CNN
%
opts.f = 1/100;

% Layer 1: Convolution Layer, 5x5 kernel, 20 features
net.layers = {};
l1Pad = 0;
if opts.useCropping
    l1Pad = 2;
end
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 1, 20, 'single'), zeros(1, 20, 'single')}}, ...
                             'stride', 1, ...
                             'pad', l1Pad);

% Layer 2: Max Pooling Layer, 2x2 window
net.layers{end + 1} = struct('type', 'pool', ...
                             'method', 'max', ...
                             'pool', [2 2], ...
                             'stride', 2, ...
                             'pad', 0);
                         
% Layer 3: Convolution Layer, 5x5 kernel, 50 features
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 20, 50, 'single'), zeros(1, 50, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
                         
% Layer 4: Max Pooling Layer, 2x2 window
net.layers{end + 1} = struct('type', 'pool', ...
                             'method', 'max', ...
                             'pool', [2 2], ...
                             'stride', 2, ...
                             'pad', 0);
                         
% Layer 5: Convolution Layer, 4x4 kernel, 500 features
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(4, 4, 50, 500, 'single'), zeros(1, 500, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
                         
% Layer 6: ReLU function
net.layers{end + 1} = struct('type', 'relu');


% Layer 7: Convolution Layer, 1x1 kernel, 10 features
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(1, 1, 500, 10, 'single'), zeros(1, 10, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
                         
% Layer 8: Softmax
net.layers{end + 1} = struct('type', 'softmaxloss');

if opts.useBnorm
    net = insertBnorm(net, 1);
    net = insertBnorm(net, 4);
    net = insertBnorm(net, 7);
end
end