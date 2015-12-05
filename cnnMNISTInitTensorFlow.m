function net = cnnMNISTInitTensorFlow(opts)
% http://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
opts.f = 1/100;

% Layer 1: Convolution Layer, 5x5 kernel, 32 features
net.layers = {};
l1Pad = 0;
if opts.useCropping
    l1Pad = 2;
end
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 1, 32, 'single'), zeros(1, 32, 'single')}}, ...
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

% Layer 4: Convolution Layer, 5x5 kernel, 64 features
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(5, 5, 32, 64, 'single'), zeros(1, 64, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);

% Layer 5: ReLU function
net.layers{end + 1} = struct('type', 'relu');

% Layer 6: Max Pooling Layer, 2x2 window
net.layers{end + 1} = struct('type', 'pool', ...
                             'method', 'max', ...
                             'pool', [2 2], ...
                             'stride', 2, ...
                             'pad', 0);
                         
% Layer 7: Convolution Layer, 4x4 kernel
net.layers{end + 1} = struct('type', 'conv', ...
                             'weights', {{opts.f * randn(4, 4, 64, 10, 'single'), zeros(1, 10, 'single')}}, ...
                             'stride', 1, ...
                             'pad', 0);
                         
% Layer 8: Dropout, 50%
net.layers{end + 1} = struct('type', 'dropout', ...
                             'rate', 0.5);

% Layer 9: Softmax
net.layers{end + 1} = struct('type', 'softmaxloss');

if opts.useBnorm
    net = insertBnorm(net, 1);
    net = insertBnorm(net, 5);
end

end
