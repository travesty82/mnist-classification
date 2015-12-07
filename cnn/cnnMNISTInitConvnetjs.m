function net = cnnMNISTInitConvnetjs(opts)
% CNNMNISTINITMATCONVNET - Initializes a CNN based on the CNN used by the
% MNIST example in the ConvNetJS project:
% http://cs.stanford.edu/people/karpathy/convnetjs/
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

% Layer 1: Convolution Layer, 5x5 kernel, 8 features
net.layers = {};
l1Pad = 0;
if opts.useCropping
    l1Pad = 2;
end
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 1, 8, 'single'), zeros(1, 8, 'single')}}, ...
                             'stride', 1, ...
                             'pad', l1Pad);
                         
% Layer 2: ReLU function
net.layers{end + 1} = struct('type', 'relu');

% Layer 3: Max Pooling Layer, 2x2 window
net.layers{end + 1} = struct('type', 'pool', ...
                             'method', 'max', ...
                             'pool', [2 2], ...
                             'stride', 2, ...
                             'pad', 0);

% Layer 4: Convolution Layer, 5x5 kernel, 16 features
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 8, 16, 'single'), zeros(1, 16, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 2);

% Layer 5: ReLU function
net.layers{end + 1} = struct('type', 'relu');

% Layer 6: Max Pooling Layer, 3x3 window
net.layers{end + 1} = struct('type', 'pool', ...
                             'method', 'max', ...
                             'pool', [3 3], ...
                             'stride', 3, ...
                             'pad', 0);

% Layer 7: Convolution Layer, 4x4 kernel
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(4, 4, 16, 10, 'single'), zeros(1, 10, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
                         
% Layer 8: Softmax
net.layers{end + 1} = struct('type', 'softmaxloss');

if opts.useBnorm
    net = insertBnorm(net, 1);
    net = insertBnorm(net, 5);
end

end
