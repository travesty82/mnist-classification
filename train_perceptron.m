function w = train_perceptron(x, t, learningRate, varargin)
% TRAIN_PERCEPTRON Trains a perceptron.
%   W = TRAIN_PERCEPTRON(X, T, LEARNINGRATE) Trains a perceptron with
%   training patterns X and target outputs T at a rate of LEARNINGRATE.
%   Training patterns are specified row-wise.
%
%   Other options:
%   TRANSFERFN - The transfer function to use to determine perceptron
%   output. Defaults to UNIPOLAR_TFN (the unipolar hard-limiting function)
%   NUMITER - The maximum number of iterations (epochs) to train for. If
%   this parameter is not specified, training will continue until the
%   outputs of the perceptron for the specified training patterns match the
%   target outputs. In the case that the problem is not linearly separable,
%   not specifying a value for this option will result in the algorithm not
%   terminating. The default value is 0 (unlimited iterations).
%   RANDOMIZE - If this is `true`, the initial weights are randomized to
%   numbers between 0 and 1. Otherwise, the weights are initialized to 0.
%   The default value is `false`.
p = inputParser;
addRequired(p, 'x');
addRequired(p, 't');
addRequired(p, 'learningRate', @isnumeric);
addOptional(p, 'transferFn', @unipolar_tfn, @(f) isa(f, 'function_handle'));
addParameter(p, 'numIter', 0, @isnumeric);
addParameter(p, 'randomize', false, @isboolean);
parse(p, x, t, learningRate, varargin{:});

[count, dim] = size(x);
assert(isvector(t), 'Target must be a vector.');
assert(count == length(t), 'Number of targets must be equivalent to number of inputs');

if p.Results.randomize
    w = rand(1, dim);
else
    w = zeros(1, dim);
end

numIter = p.Results.numIter;
transferFn = p.Results.transferFn;
iter = 0;
while true
    for i = 1:count
        xi = x(i, :);
        oi = feval(transferFn, dot(xi, w));
        error = t(i) - oi;
        for j = 1:dim
            w(j) = w(j) + (learningRate * error * xi(j));
        end
    end
    iter = iter + 1;
    
    % If a number of iterations wasn't specified, terminate
    % the learning algorithm when the output of the perceptron
    % matches the target output.
    %
    % In the case that a number of iterations was not specified
    % and the problem is not linearly separable, the algorithm
    % will not terminate.
    if numIter == 0
        o = zeros(1, count);
        for i = 1:count
            o(i) = feval(transferFn, dot(x(i, :), w));
        end
        if isequal(t, o)
            break
        end
    elseif iter >= numIter
        break
    end
end
end
