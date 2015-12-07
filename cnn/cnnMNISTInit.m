function net = cnnMNISTInit(id, options)
% CNNMNISTINIT - Initializes the neural network architecture with the
% specified ID.
%
% INPUTS
%   id - The ID of the neural network to initialize.
%       1 - ConvNetJS
%       2 - TensorFlow
%       3 - MatConvNet
%   options - Options to pass to the neural network initialization function
% OUTPUTS
%   net - Initialized CNN
%
if id == 1
    net = cnnMNISTInitConvnetjs(options);
elseif id == 2
    net = cnnMNISTInitTensorFlow(options);
elseif id == 3
    net = cnnMNISTInitMatconvnet(options);
end
end
