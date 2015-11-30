function predictions = predictWithRandomCrop(net, images, labels, numClasses, cx, cy, n)
predictions = zeros(n, numClasses, size(labels, 2));
net.layers{end}.class = labels;
for i = 1:n
    croppedImages = single(cropImageBatchRandom(images, cx, cy));
    res = vl_simplenn(net, croppedImages);
    predictions(i, :, :) = squeeze(res(end - 1).x);
end
predictions = mean(predictions, 1);
[~, predictions] = sort(predictions, 2, 'descend');
predictions = reshape(predictions(:, 1, :), 1, []);
end
