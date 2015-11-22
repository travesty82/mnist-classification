function [w, b] = mlpInit(layers)
% MLPINIT - Initializes a multi layer perceptron neural network with random
% weights and biases.
%
%   [W, B] = MLPINIT(LAYERS)
%
%   INPUTS
%       LAYERS - A vector with the number of neurons per layer. There must
%       be at least 2 layers (input and output).
%
%   OUTPUTS
%       W - A cell array of weights per layer. w{i} is initialized to a 
%       matrix such that value of w{i}(j, k) is the weight of the connection 
%       from neuron k in the (i-1)th layer to neuron j in the ith layer.
%       B - A cell array of bias values per layer. b{i} is initialized to a 
%       vector such that there is one bias value per neuron in layer i.
layerCount = length(layers);
assert(layerCount >= 2, 'There must be at least 2 layers');

w = cell(1, layerCount);
b = cell(1, layerCount);
for i = 2:layerCount
    w{i} = rand(layers(i), layers(i - 1));
    b{i} = rand(layers(i), 1);
end
end
