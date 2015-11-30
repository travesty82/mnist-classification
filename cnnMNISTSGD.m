function net = cnnMNISTSGD(images, labels, getBatch, varargin)
    % ########################
    % CONFIGURATION PARAMETERS
    % ########################
    
    opts.dataDir = 'mnist';
    opts.batchSize = 256; % Batch size must be divisible by 32.
    opts.numEpochs = 20;
    opts.learningRate = 0.001;
    opts.weightDecay = 0.0005;
    opts.momentum = 0.9;
    opts = vl_argparse(opts, varargin);
    
    % ########################
    % INITIALIZATION
    % ########################
    
    net = cnnMNISTInit();
    numBatches = ceil(size(labels, 2) / opts.batchSize);
    
    % ########################
    % TRAINING
    % ########################
    
    for l = 1:numel(net.layers)
        if ~isfield(net.layers{l}, 'weights') 
            continue;
        end
        numWeights = numel(net.layers{l}.weights);
        for j = 1:numWeights
            net.layers{l}.momentum{j} = zeros(size(net.layers{l}.weights{j}), 'single') ;
        end
    end
    
    res = [];
    for e = 1:opts.numEpochs
        for b = 1:numBatches
            fprintf('Batch %d/%d (epoch %d/%d)\n', b, numBatches, e, opts.numEpochs);
            [imb, lb] = getBatch(images, labels, opts.batchSize, b);
            net.layers{end}.class = lb;
            res = vl_simplenn(net, imb, single(1), res);
        end
        
        for l = numel(net.layers):-1:1
            if ~isfield(net.layers{l}, 'weights') 
                continue;
            end
            for j = 1:numel(res(l).dzdw)
                net.layers{l}.momentum{j} = ...
                    opts.momentum * net.layers{l}.momentum{j} ...
                    - opts.weightDecay * net.layers{l}.weights{j} ...
                    - (1 / opts.batchSize) * res(l).dzdw{j};
                net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
                    + opts.learningRate * net.layers{l}.momentum{j};
            end
        end
    end
end
