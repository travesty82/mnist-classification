function mlp = mlp_train(hiddenLayers, xs, ts, learningRate, varargin)
% MLP_TRAIN Trains a multi-layer perceptron.
%   MLP = MLP_TRAIN(HIDDENLAYERS, XS, TS, LEARNINGRATE) Trains a
%   multi-layer perceptron using inputs XS (specified row-wise), target
%   outputs TS (specified row wise), at a rate of LEARNINGRATE.
%   HIDDENLAYERS specifies the number of neurons in each hidden layer, e.g.
%   [4 5] for 2 hidden layers with 4 neurons in the first hidden layer and
%   5 neurons in the second hidden layer.
%
%   OPTIONS
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
%       * https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
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
nabla_b = cell(1, layerCount - 1);
nabla_w = cell(1, layerCount - 1);
delta_nabla_b = cell(1, layerCount - 1);
delta_nabla_w = cell(1, layerCount - 1);
for i = 1:layerCount - 1
    % b{i} is initialized to a vector such that there is one bias value
    % per neuron in layer i. w{i} is initialized to a matrix such that
    % value of w{i}(k, j) is the weight of the connection from neuron j to
    % neuron k, where k is a neuron that exists in layer i.
    if p.Results.randomize
        w{i} = genRandom(layers(i), layers(i + 1));
        b{i} = genRandom(1, layers(i));
    else
        w{i} = zeros(layers(i), layers(i + 1));
        b{i} = zeros(1, layers(i));
    end
    % gradient cost descent
    nabla_w{i} = zeros(layers(i), layers(i + 1));
    nabla_b{i} = zeros(1, layers(i));
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

        % FEED FORWARD: Calculate outputs
        % ============
        %
        % o is a cell array that contains the output values for each
        % neuron of each layer. o{i} is the vector containing the output
        % values for all neurons on layer i.
        o = cell(1, layerCount);
        zs = cell(1, layerCount);
        % "Output" for input layer is just the input vector.
        o{1} = x';
        % For hidden and output layers...
        % w{1} is weight from 1 to 2
        % b{1} is bias from 1 to 2
        % hence update layer i + 1 with values from i.
        
        % Extra Back Prop
        % gradient for the cost function
 

        
        % forward and back prop can be done in one loop
        
        for layer = 1:layerCount - 1
            zs{layer} = b{layer} + rowDot(w{layer},o{layer})';
            o{layer + 1} = f(zs{layer});
            
            % zero out delta while we are at it
            delta_nabla_w{layer} = zeros(layers(layer), layers(layer + 1));
            delta_nabla_b{layer} = zeros(1, layers(layer));
        end
        delta = (o{end} - t) .* df(zs{end - 1});
        delta_nabla_w{end} = dot(delta,o{end - 2});
        delta_nabla_b{end} = delta;
        
        % BACK PROPAGATION: Calculate deltas
        % ================
        %
        % Calculate delta for hidden layers
        % Start at the 2nd last layer to 2nd layer
        % reverse direction
        for layer = (layerCount - 2):-1:1
            z = zs{layer};
            sp = df(z);
            delta = rowDot(w{layer + 1},delta)' .* sp;
            delta_nabla_b{layer} = delta;
            delta_nabla_w{layer} = dot(o{layer},delta)';
        end
        
        % Update cumulative error
        cumulativeError = cumulativeError + (1/2 * length(xs)) * sum(delta .^ 2);
        disp(cumulativeError);

        % WEIGHT AND BIAS UPDATE
        % ======================
        
        for layer = 1:layerCount-1
            nabla_w{layer} = nabla_w{layer} + delta_nabla_w{layer};
            nabla_b{layer} = nabla_b{layer} + delta_nabla_b{layer};
            
            w{layer} = w{layer} - ((learningRate/length(xs)) * nabla_w{layer});
            b{layer} = b{layer} - ((learningRate/length(xs)) * nabla_b{layer});
        end
        
    end
    
    % Keep training until the specified number of iterations or a specified
    % error threshold is reached.
    iter = iter + 1;
    if (numIter > 0 && iter >= numIter )%  || (cumulativeError <= threshold)
        break
    end
end
mlp = struct();
mlp.weights = w;
mlp.bias = b;
mlp.layers = layers;
mlp.activationFn = f;
end