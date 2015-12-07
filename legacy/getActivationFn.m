function [f, df] = getActivationFn(fname)
% GETACTIVATIONFN - Returns an activation function and its derivative.
%
%   [F, DF] = GETACTIVATIONFN(FNAME)
%
%   INPUTS
%       FNAME - The name of the activation function. Must be one of either
%       'logsig' or 'tansig'
%
%   OUTPUTS
%       F - The activation function
%       DF - The derivative of the activation function
%
if strcmp(fname, 'logsig')
    f = @logsig;
    df = @(x) dlogsig(x, logsig(x));
elseif strcmp(fname, 'tansig')
    f = @tansig;
    df = @(x) dtansig(x, tansig(x));
else
    assert(false, 'Invalid activation function.');
end
end
