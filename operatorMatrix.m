function [ m ] = operatorMatrix( A,B,operator )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if size(A) == size(B)
    % same size perform normal addition
    m = operator(A,B);
elseif size(A,1) < size(B,1) && size(A,2) == size(B,2)
    % copy col wise a
    m = operator(padarray(A,size(B) - size(A),'replicate','post'),B);
elseif size(A,1) > size(B,1) && size(A,2) == size(B,2)
     % copy col wise b
    m = operator(A,padarray(B,size(A) - size(B),'replicate','post'));
elseif size(A,2) < size(B,2) && size(A,1) == size(B,1)
    % copy col wise a
    m = operator(padarray(A,size(B) - size(A),'replicate','post'),B);
elseif size(A,2) > size(B,2) && size(A,1) == size(B,1)
    % copy col wise b
    m = operator(A,padarray(B,size(A) - size(B),'replicate','post'));
elseif size(A') == size(B)
    m = operator(A',B);
else
    error('Error. \n ith or jth dimension must match');
end

end

