function y = mlpPredict(mlp, x)
% MLP_PREDICT Predicts the output of a trained multi-layer perceptron for a
% given input vector.
%
%   Y = MLP_PREDICT(MLP, X) Y is the predicted output for the input vector
%   X by the multi-layer perceptron MLP.
%
layerCount = length(mlp.layers);
f = mlp.activationFn;
a = cell(1, layerCount);
a{1} = x';
for i = 2:layerCount
    a{i} = f(mlp.weights{i} * a{i - 1} + mlp.bias{i});
end
y = a{end};
end
