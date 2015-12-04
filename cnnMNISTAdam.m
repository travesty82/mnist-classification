function net = cnnMNISTAdam(images, labels, getBatch, varargin)
    % ADAM Optimization algorithm as described in:
    % * http://arxiv.org/pdf/1412.6980v8.pdf (Original Paper)
    % * http://caffe.berkeleyvision.org/tutorial/solver.html (Caffe
    % Implementation)

    % ########################
    % CONFIGURATION PARAMETERS
    % ########################
    
    opts.batchSize = 32; % Batch size must be divisible by 32.
    opts.numEpochs = 20;
    opts.learningRate = 0.001;
    opts.momentum1 = 0.9;
    opts.momentum2 = 0.999;
    opts.epsilon = 1e-8;
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
            net.layers{l}.moment1{j} = zeros(size(net.layers{l}.weights{j}), 'single');
            net.layers{l}.moment2{j} = zeros(size(net.layers{l}.weights{j}), 'single');
        end
    end
    
    res = [];
    for e = 1:opts.numEpochs
        c = (sqrt(1 - (opts.momentum2 .^ e)) / (1 - (opts.momentum1 .^ e)));
        for b = 1:numBatches
            [imb, lb] = getBatch(images, labels, opts.batchSize, b);
            net.layers{end}.class = lb;
            res = vl_simplenn(net, imb, single(1), res);
            fprintf('Batch %d/%d (epoch %d/%d)\n', b, numBatches, e, opts.numEpochs);
            
            for l = numel(net.layers):-1:1
                if ~isfield(net.layers{l}, 'weights') 
                    continue;
                end
                for j = 1:numel(res(l).dzdw)
                    net.layers{l}.moment1{j} = ...
                        opts.momentum1 * net.layers{l}.moment1{j} ...
                        + (1 - opts.momentum1) * res(l).dzdw{j};
                    net.layers{l}.moment2{j} = ...
                        opts.momentum2 * net.layers{l}.moment2{j} ...
                        + (1 - opts.momentum2) * (res(l).dzdw{j} .^ 2);
                    net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
                        - opts.learningRate * c...
                        * net.layers{l}.moment1{j} ./ (sqrt(net.layers{l}.moment2{j}) + opts.epsilon);
                end
            end
        end
    end
end
