function y = unipolar_tfn(x)
% UNIPOLAR_TFN Unipolar hard-limit transfer function.
%   Y = UNIPOLAR_TFN(X) applies the transfer function to X.

if x >= 0
    y = 1;
else
    y = 0;
end
end