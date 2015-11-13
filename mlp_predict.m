function y = mlp_predict(mlp, x)
% MLP_PREDICT Predicts the output of a trained multi-layer perceptron for a
% given input vector.
%
%   Y = MLP_PREDICT(MLP, X) Y is the predicted output for the input vector
%   X by the multi-layer perceptron MLP.
%
    layerCount = length(mlp.layers);
    o = cell(1, layerCount);
    o{1} = x;
    for i = 2:layerCount
        o{i} = mlp.activationFn(mlp.bias{i} + sum(mlp.weights{i} * o{i - 1}'));
    end
    y = o{end};
end
