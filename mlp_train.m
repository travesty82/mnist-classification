function mlp = mlp_train(hiddenLayers, xs, ts, learningRate, varargin)
%
% REFERENCES
%   * http://ml.informatik.uni-freiburg.de/_media/documents/teaching/ss09/ml/mlps.pdf
%   * http://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
%   
p = inputParser;
addRequired(p, 'hiddenLayers');
addRequired(p, 'x');
addRequired(p, 't');
addRequired(p, 'learningRate', @isnumeric);
addOptional(p, 'activationFn', 'logsig', @ischar);
addParameter(p, 'numIter', 0, @isnumeric);
addParameter(p, 'randomize', true, @isboolean);
parse(p, hiddenLayers, xs, ts, learningRate, varargin{:});

assert(size(xs, 1) == size(ts, 1), 'Number of input vectors and number of target outputs must be equal');

layers = [size(xs, 2) hiddenLayers size(ts, 2)];
layerCount = length(layers);
activationFn = p.Results.activationFn;
if strcmp(activationFn, 'logsig')
    f = @logsig;
elseif strcmp(activationFn, 'tansig')
    f = @tansig;
else
    assert(false, 'Invalid activation function.');
end

% Initialize the weights and biases to either zero or small random
% values in the range [-0.25, 0.25] depending on whether the
% `randomize` option is specified.
gen_random = @(z) (rand(1, z) * 0.5) - 0.25;
w = cell(1, layerCount);
b = cell(1, layerCount);
for i = 1:layerCount
    neuronCount = layers(i);
    if p.Results.randomize
        w{i} = gen_random(neuronCount);
        b{i} = gen_random(neuronCount);
    else
        w{i} = zeros(1, neuronCount);
        b{i} = zeros(1, neuronCount);
    end
end

% Input layer should not have any bias, so set all bias
% values to zero.
b{1} = zeros(1, size(xs, 2));

% Forward propagation
for k = 1:length(xs)
    x = xs(k, :);
    a = cell(1, layerCount);
    a{1} = x;
    for i = 2:layerCount
        net = b{i} + dot(w{i - 1}, a{i - 1});
        a{i} = f(net);
    end
end


mlp = struct('weights', w, 'bias', b, 'layers', layers, 'activationFn', activationFn);
end