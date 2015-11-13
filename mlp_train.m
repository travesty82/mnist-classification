function mlp = mlp_train(hiddenLayers, xs, ts, learningRate, varargin)
% MLP_TRAIN Trains a multi-layer perceptron.
%   MPL = MPL_TRAIN(HIDDENLAYERS, XS, TS, LEARNINGRATE) Trains a
%   multi-layer perceptron using inputs XS (specified row-wise), target
%   outputs TS (specified row wise), at a rate of LEARNINGRATE.
%   HIDDENLAYERS specifies the number of neurons in each hidden layer, e.g.
%   [4 5] for 2 hidden layers with 4 neurons in the first hidden layer and
%   5 neurons in the second hidden layer.
%
%   Options:
%       ACTIVATIONFN - The activation function to use when determining
%       neuron output. Must be one of 'logsig' or 'tansig'. Default value
%       is 'logsig'
%       NUMITER - The number of iterations (epochs) to train for.
%       THRESHOLD - The cumulative error threshold. Training will continue
%       until the cumulative error is less than or equal to this threshold.
%       If both NUMITER and THRESHOLD are specified, training will stop
%       when either one of the conditions are reached (whichever comes
%       first)
%       RANDOMIZE - If this is `true`, the initial weights and biases will
%       be initialized randomly to small values in the range [-0.25, 0.25].
%       If this is `false`, all weights and biases will be initialized to
%       zero. Default value is `true`.
%
%   REFERENCES
%       * http://ml.informatik.uni-freiburg.de/_media/documents/teaching/ss09/ml/mlps.pdf
%       * http://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
%       * http://rolisz.ro/2013/04/18/neural-networks-in-python/
%       * https://www.willamette.edu/~gorr/classes/cs449/backprop.html
%       * http://sydney.edu.au/engineering/it/~comp4302/ann4-3s.pdf
%
p = inputParser;
addRequired(p, 'hiddenLayers');
addRequired(p, 'x');
addRequired(p, 't');
addRequired(p, 'learningRate', @isnumeric);
addOptional(p, 'activationFn', 'logsig', @ischar);
addParameter(p, 'numIter', 0, @isnumeric);
addParameter(p, 'threshold', 0, @isnumeric);
addParameter(p, 'randomize', true, @isboolean);
parse(p, hiddenLayers, xs, ts, learningRate, varargin{:});

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
genRandom = @(m, n) (rand(m, n) * 0.5) - 0.25;
w = cell(1, layerCount);
b = cell(1, layerCount);
for i = 2:layerCount
    if p.Results.randomize
        w{i} = genRandom(layers(i), layers(i - 1));
        b{i} = genRandom(1, layers(i));
    else
        w{i} = zeros(layers(i), layers(i - 1));
        b{i} = zeros(1, layers(i));
    end
end

numIter = p.Results.numIter;
threshold = p.Results.threshold;
iter = 0;
while true
    cumulativeError = 0;
    % One epoch of training
    for s = 1:length(xs)
        x = xs(s, :);
        t = ts(s, :);

        % FEED FORWARD
        % ============
        %
        o = cell(1, layerCount);
        % "Output" for input layer is just the input vector.
        o{1} = x;
        % For hidden and output layers...
        for i = 2:layerCount
            o{i} = f(b{i} + sum(w{i} * o{i - 1}'));
        end

        % BACK PROPAGATION
        % ================
        %
        d = cell(1, layerCount);
        error = t - o{end};
        cumulativeError = cumulativeError + (1/2) .* sum(error .^ 2);
        d{end} = error .* df(o{end});
        for i = (layerCount - 1):-1:2
            d{i} = df(o{i}) .* sum(sum(w{i + 1}, 2) .* d{i + 1}');
        end

        % WEIGHT AND BIAS UPDATE
        % ======================
        %
        for i = 2:layerCount
            w{i} = w{i} + (learningRate * d{i}' * o{i - 1});
            b{i} = b{i} + (learningRate * d{i});
        end
    end
    
    % Keep training until the specified number of iterations or a specified
    % error threshold is reached.
    iter = iter + 1;
    if (numIter > 0 && iter >= numIter) || (cumulativeError <= threshold)
        break
    end
end
mlp = struct();
mlp.weights = w;
mlp.bias = b;
mlp.layers = layers;
mlp.activationFn = f;
end