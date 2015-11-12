function y = bipolar_tfn(x)
% BIPOLAR_TFN Bipolar hard-limit transfer function.
%   Y = BIPOLAR_TFN(X) applies the transfer function to X.

if x >= 0
    y = 1;
else
    y = -1;
end
end
