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


for xi = 1:length(xs)
    x = xs(xi, :);
    
    % FEED FORWARD STEP
    % ===================
    %
    o = cell(1, layerCount);
    % "Output" for input layer is just the input vector.
    o{1} = x;
    % For hidden and output layers...
    for i = 2:layerCount
        n = layers(i);
        net = zeros(1, n);
        for j = 1:n
            net(j) = b{i}(j) + dot(w{i}(j, :), o{i - 1});
        end
        o{i} = f(net);
    end
    display(o);
end

mlp = struct('weights', w, 'bias', b, 'layers', layers, 'activationFn', activationFn);
end