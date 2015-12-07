function net = insertBnorm(net, l)
% INSERTBNORM - Inserts a batch normalization layer into a CNN. Copied from
% matconvnet/examples/cnn_mnist_init.m
%
% INPUTS
%   net - The CNN to insert the layer into
%   l - The index of the convolution layer after which to insert the 
%   normalization layer.
%
% OUTPUTS
%   net - The net with the batch normalization layer inserted.
% 
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
end
