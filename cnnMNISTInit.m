function [images, labels, net] = cnnMNISTInit(dataDir, isTraining)
    % ########################
    % DATA LOADING
    % ########################
    
    if isTraining
        images = loadMNISTImages(fullfile(dataDir, 'train-images-idx3-ubyte'));
        labels = loadMNISTLabels(fullfile(dataDir, 'train-labels-idx1-ubyte'))';
    else
        images = loadMNISTImages(fullfile(dataDir, 't10k-images-idx3-ubyte'));
        labels = loadMNISTLabels(fullfile(dataDir, 't10k-labels-idx1-ubyte'))';
    end
    
    imageDim = size(images, 1);
    images = single(reshape(images, imageDim, imageDim, 1, []));
    
    % ########################
    % NETWORK CONFIGURATION
    % ########################
    
    % Scaling factor for randomly generated values
    f = 1 / 100;
    
    % Layer 1: Convolution Layer, 5x5 kernel, 32 features
    net.layers = {};
    net.layers{end + 1} = struct('type', 'conv', ...
                                 'weights', {{f * randn(5, 5, 1, 32, 'single'), zeros(1, 32, 'single')}}, ...
                                 'stride', 1, ...
                                 'pad', 0);
    % Layer 2: ReLU function
    net.layers{end + 1} = struct('type', 'relu');
    
    % Layer 3: Max Pooling Layer, 2x2 kernel
    net.layers{end + 1} = struct('type', 'pool', ...
                                 'method', 'max', ...
                                 'pool', [2 2], ...
                                 'stride', 2, ...
                                 'pad', 0);
                             
    % Layer 4: Convolution Layer, 5x5 kernel, 64 features
    net.layers{end + 1} = struct('type', 'conv', ...
                                 'weights', {{f * randn(5, 5, 32, 64, 'single'), zeros(1, 64, 'single')}}, ...
                                 'stride', 1, ...
                                 'pad', 0);
                             
    % Layer 5: ReLU function
    net.layers{end + 1} = struct('type', 'relu');
    
    % Layer 6: Max Pooling Layer, 2x2 kernel
    net.layers{end + 1} = struct('type', 'pool', ...
                                 'method', 'max', ...
                                 'pool', [2 2], ...
                                 'stride', 2, ...
                                 'pad', 0);
    
    % Layer 7: Fully Connected Layer, 7x7 kernel, 10 outputs
    net.layers{end + 1} = struct('type', 'conv', ...
                                 'weights', {{f * randn(1, 1, 64, 10, 'single'), zeros(1, 10, 'single')}}, ...
                                 'stride', 1, ...
                                 'pad', 0);
                             
    % Layer 8: Dropout (only if training)
    if isTraining
        net.layers{end + 1} = struct('type', 'dropout', 'rate', 0.5);
    end
    
    % Layer 9: Softmax
    net.layers{end + 1} = struct('type', 'softmax');
end
