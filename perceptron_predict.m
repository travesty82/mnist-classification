function y = perceptron_predict(p, x)
% PERCEPTRON_PREDICT Predicts the output for a given input vector using a
% perceptron trained using PERCEPTRON_TRAIN
%
%   Y = PERCEPTRON_PREDICT(P, X) Y is the output value for input vector X,
%   predicted using perceptron P.
    assert(length(p.weights) == size(x, 2), 'Dimension of input does not match the dimension of the perceptron');
    y = feval(p.transferFn, dot(x, p.weights));
end
