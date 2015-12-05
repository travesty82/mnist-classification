function net = cnnMNISTSGD(images, labels, getBatch, varargin)
    % ########################
    % CONFIGURATION PARAMETERS
    % ########################
    
    opts.batchSize = 32; % Batch size must be divisible by 32.
    opts.numEpochs = 20;
    opts.learningRate = 0.01;
    opts.weightDecay = 0.001;
    opts.momentum = [0.5,0.85:(0.92-0.85)/opts.numEpochs-1:0.92];
    opts.net = cnnMNISTInit();
    opts = vl_argparse(opts, varargin);
    
    % ########################
    % INITIALIZATION
    % ########################
    
    net = opts.net;
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
            [imb, lb] = getBatch(images, labels, opts.batchSize, b);
            net.layers{end}.class = lb;
            res = vl_simplenn(net, imb, double(1), res);
            fprintf('Batch %d/%d (epoch %d/%d)\n', b, numBatches, e, opts.numEpochs);
            
            for l = numel(net.layers):-1:1
                if ~isfield(net.layers{l}, 'weights') 
                    continue;
                end
                for j = 1:numel(res(l).dzdw)
                    net.layers{l}.momentum{j} = ...
                        opts.momentum(e) * net.layers{l}.momentum{j} ...
                        - opts.weightDecay * net.layers{l}.weights{j} ...
                        - (1 / opts.batchSize) * res(l).dzdw{j};
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
                        + opts.learningRate * net.layers{l}.momentum{j};
                end
            end
        end
    end
end
