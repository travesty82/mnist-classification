function net = cnnTrainSGD(images, labels, getBatch, varargin)
%   CNNTRAINSGD - Trains a CNN using stochastic gradient descent with
%   momentum and weight decay
%
%   INPUTS
%       images - Matrix of training images. M x N x 1 x (number of images)
%       labels - Matrix of training labels. 1 x (number of images)
%       getBatch - Function handle for getting the next batch of training
%       data. Should be in the form getBatch(images, labels, batchSize, batchNumber)
%       varargin - Additional optional configuration parameters
%           batchSize - Number of training samples per batch
%           numEpochs - Number of training epochs
%           learningRate - Learning rate for SGD
%           momentum - Momentum for SGD
%           weightDecay - Weight decay rate for SGD
%           netID - ID of neural network type {1, 2, 3}
%           useBnorm - Whether to use batch normalization
%           useCropping - Whether images have been preprocessed by cropping
%           random windows (data augmentation)
%           gpus - IDs of GPU devices to use
%
%   OUTPUTS
%       net - Trained CNN
%
    % ########################
    % CONFIGURATION PARAMETERS
    % ########################
    
    opts.batchSize = 32;
    opts.numEpochs = 20;
    opts.learningRate = 0.01;
    opts.weightDecay = 0.001;
    opts.momentum = 0.9;
    opts.netID = 1;
    opts.useBnorm = false;
    opts.useCropping = false;
    opts.gpus = [] ; % which GPU devices to use (none, one, or more)
    opts = vl_argparse(opts, varargin);
    % setup GPUs
    numGpus = numel(opts.gpus) ;
    if numGpus > 1
      if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(opts.gpus(labindex)), end
      end
    elseif numGpus == 1
      gpuDevice(opts.gpus)
    end
    
    % ########################
    % INITIALIZATION
    % ########################
    
    net_cpu = cnnMNISTInit(opts.netID, opts);
    numBatches = ceil(size(labels, 2) / opts.batchSize);

   % move CNN to GPU as needed
   numGpus = numel(opts.gpus) ;
   if numGpus >= 1
      fprintf('Using GPU for NN\n');
      images = gpuArray(images);
      labels = gpuArray(labels);
      net = vl_simplenn_move(net_cpu, 'gpu') ;
   else
      net = net_cpu ;
      net_cpu = [] ;
   end
    
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
            net = vl_simplenn_tidy(net);
            res = vl_simplenn(net, imb, single(1), res);
            fprintf('Batch %d/%d (epoch %d/%d)\n', b, numBatches, e, opts.numEpochs);
            
            for l = numel(net.layers):-1:1
                if ~isfield(net.layers{l}, 'weights') 
                    continue;
                end
                for j = 1:numel(res(l).dzdw)
                    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                        % special case for learning moments
                        thisLR = net.layers{l}.learningRate(j);
                        net.layers{l}.weights{j} = ...
                            (1 - thisLR) * net.layers{l}.weights{j} + ...
                            (thisLR/opts.batchSize) * res(l).dzdw{j};
                    else                
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
    end
   numGpus = numel(opts.gpus) ;
   if numGpus >= 1
      % Move back to cpu
      net = vl_simplenn_move(net, 'cpu') ;
   end
end
