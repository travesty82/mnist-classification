function products = rowDot( w,o )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    length = size(w,2);
    products = zeros(length,1);
    
if length < 2
    products(1) = dot(w,o);
    return;
end
    
if size(o) == [1,1]
    products = w .* o;
    return;
end

for i = 1:length
    products(i) = dot(w(i,:),o);
end



end

