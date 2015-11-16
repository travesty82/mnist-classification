function products = rowDotBias( w,o,bias )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    length = max([size(w);size(o)]);
    products = zeros(size(bias,2):length);
    


if(size(w,1) > size(o,1))
    for i = 1:length(1)
        products(i,:) = dot(w(i,:),o) + bias;
    end
else
    for i = 1:length(2)
        products(i) = dot(o(:,i),w) + bias;
    end
end


end

