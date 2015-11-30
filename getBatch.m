function [imb, lb] = getBatch(images, labels, batchSize, batchNumber)
batchStart = (batchSize * (batchNumber - 1)) + 1;
batchEnd = min(batchStart + batchSize - 1, size(labels, 2));
imb = images(:, :, :, batchStart:batchEnd);
lb = labels(:, batchStart:batchEnd);
end
