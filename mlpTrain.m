function [mlp, mse] = mlpTrain(hiddenLayers, xs, ts, learningRate, numIter, varargin)
% MLPTRAIN Trains a multi-layer perceptron.
%   [MLP. MSE] = MLPTRAIN(HIDDENLAYERS, XS, TS, LEARNINGRATE, NUMITER) 
%   Trains a multi-layer perceptron using inputs XS (specified row-wise),
%   target outputs TS (specified row wise), at a rate of LEARNINGRATE.
%   HIDDENLAYERS specifies the number of neurons in each hidden layer, e.g.
%   [4 5] for 2 hidden layers with 4 neurons in the first hidden layer and
%   5 neurons in the second hidden layer. NUMITER specifies the maximum
%   number of iterations to run training for. MSE is the mean squared error
%   vector, with size 1xnumIter.
%
%   OPTIONS
%       ACTIVATIONFN - The activation function to use when determining
%       neuron output. Must be one of 'logsig' or 'tansig'. Default value
%       is 'logsig'
%       THRESHOLD - The mean squared error threshold. Training will stop
%       before the maximum number of iterations have been reached if the
%       mean squared error is less than or equal to this threshold.
%       RANDOMIZE - If this is `true`, the initial weights and biases will
%       be initialized randomly to small values in the range [-0.25, 0.25].
%       If this is `false`, all weights and biases will be initialized to
%       zero. Default value is `true`.
%       MOMENTUM - Momentum coefficient that adds speed to the training of
%       the network by adding a part of the previous epoch's weight update
%       to the current epoch's weight update.
%
%   REFERENCES
%       * http://ml.informatik.uni-freiburg.de/_media/documents/teaching/ss09/ml/mlps.pdf
%       * http://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
%       * http://rolisz.ro/2013/04/18/neural-networks-in-python/
%       * https://www.willamette.edu/~gorr/classes/cs449/backprop.html
%       * http://sydney.edu.au/engineering/it/~comp4302/ann4-3s.pdf
%       * http://neuralnetworksanddeeplearning.com/chap2.html
%
p = inputParser;
addRequired(p, 'hiddenLayers');
addRequired(p, 'xs');
addRequired(p, 'ts');
addRequired(p, 'learningRate', @isnumeric);
addRequired(p, 'numIter', @isnumeric);
addOptional(p, 'activationFn', 'logsig', @ischar);
addParameter(p, 'threshold', 0, @isnumeric);
addParameter(p, 'randomize', true, @isboolean);
addParameter(p, 'momentum', 0, @isnumeric);
parse(p, hiddenLayers, xs, ts, learningRate, numIter, varargin{:});

assert(size(xs, 1) == size(ts, 1), 'Number of input vectors and number of target outputs must be equal');

layers = [size(xs, 2) hiddenLayers size(ts, 2)];
layerCount = length(layers);
activationFn = p.Results.activationFn;
if strcmp(activationFn, 'logsig')
    f = @logsig;
    df = @(x) dlogsig(x, logsig(x));
elseif strcmp(activationFn, 'tansig')
    f = @tansig;
    df = @(x) dtansig(x, tansig(x));
else
    assert(false, 'Invalid activation function.');
end

% Initialize the weights and biases to either zero or small random
% values in the range [-0.25, 0.25] depending on whether the
% `randomize` option is specified.
w = cell(1, layerCount);
b = cell(1, layerCount);
% Keep track of previous weights to use with momentum.
pdw = cell(1, layerCount);
pdb = cell(1, layerCount);
for i = 2:layerCount
    % b{i} is initialized to a vector such that there is one bias value
    % per neuron in layer i. w{i} is initialized to a matrix such that
    % value of w{i}(j, k) is the weight of the connection from neuron k in
    % the (i-1)th layer to neuron j in the ith layer.
    lj = layers(i);
    lk = layers(i - 1);
    if p.Results.randomize
        w{i} = rand(lj, lk);
        b{i} = rand(lj, 1);
    else
        w{i} = zeros(lj, lk);
        b{i} = zeros(lj, 1);
    end
    pdw{i} = zeros(lj, lk);
    pdb{i} = zeros(lj, 1);
end

threshold = p.Results.threshold;
momentum = p.Results.momentum;
iter = 0;
xs = xs'; ts = ts';
numSamples = size(xs, 2);
mse = zeros(1, numIter);
while true
   	mseEpoch = zeros(1, numSamples);
    % One epoch of training
    for s = randperm(numSamples)
        x = xs(:, s);
        t = ts(:, s);

        % FEED FORWARD: Calculate outputs
        % ============
        % z is a cell array that contains the weighted inputs for each
        % neuron of each layer. z{i} is the vector containing the output
        % values for all neurons on layer i.
        %
        % a is a cell array that contains the output values for each
        % neuron of each layer. a is related to z by the equation
        % a = f(z) where f is the activation function.
        z = cell(1, layerCount);
        a = cell(1, layerCount);
        a{1} = x;
        for i = 2:layerCount
            z{i} = w{i} * a{i - 1} + b{i};
            a{i} = f(z{i});
        end

        % BACK PROPAGATION: Calculate deltas
        % ================
        %
        % d is a cell array containing the deltas for each neuron of
        % each layer.
        d = cell(1, layerCount);
        error = a{end} - t;
        mseEpoch(i) = mean(error .^ 2);
        d{end} = error .* df(z{end});
        for i = (layerCount - 1):-1:2
            d{i} = (w{i + 1}' * d{i + 1}) .* df(z{i});
        end

        % WEIGHT AND BIAS UPDATE
        % ======================
        %
        dw = cell(1, layerCount);
        db = cell(1, layerCount);
        for i = 2:layerCount
            dw{i} = -(learningRate * d{i} * a{i - 1}') + (momentum .* pdw{i});
            db{i} = -(learningRate * d{i}) + (momentum .* pdb{i});
            w{i} = w{i} + dw{i};
            b{i} = b{i} + db{i};
        end
        pdw = dw;
        pdb = db;
    end
    
    % Keep training until the specified number of iterations or a specified
    % error threshold is reached.
    iter = iter + 1;
    meanMseEpoch = mean(mseEpoch);
    mse(iter) = meanMseEpoch;
    if (iter >= numIter) || (meanMseEpoch <= threshold)
        break
    end
end
mlp = struct();
mlp.weights = w;
mlp.bias = b;
mlp.layers = layers;
mlp.activationFn = f;
end
