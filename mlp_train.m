function mlp = mlp_train(hiddenLayers, xs, ts, learningRate, varargin)
%
% REFERENCES
%   * http://ml.informatik.uni-freiburg.de/_media/documents/teaching/ss09/ml/mlps.pdf
%   * http://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
%   * http://rolisz.ro/2013/04/18/neural-networks-in-python/
%   * https://www.willamette.edu/~gorr/classes/cs449/backprop.html
%   * http://sydney.edu.au/engineering/it/~comp4302/ann4-3s.pdf
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
    df = @(x) dlogsig(x, logsig(x));
elseif strcmp(activationFn, 'tansig')
    f = @tansig;
    df = @(x) dtansig(x, tansig(x));
else
    assert(false, 'Invalid activation function.');
end

rng(40);

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

% For each input vector.
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
        net = b{i} + dot(w{i}, repmat(o{i - 1}, layers(i), 1), 2)';
        o{i} = f(net);
    end
    
    % BACK PROPAGATION
    % ================
    %
    d = cell(1, layerCount);
    error = t - o{end};
    d{end} = error .* df(o{end});
    for i = (layerCount - 1):-1:2
        d{i} = df(o{i}) .* sum(w{i + 1} * d{i + 1}');
    end
end

mlp = struct('weights', w, 'bias', b, 'layers', layers, 'activationFn', activationFn);
end