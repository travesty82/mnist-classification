function croppedImages = cropImageBatchRandom(images, cx, cy)
imageCount = size(images, 4);
croppedImages = zeros(cx, cy, 1, imageCount);
for i = 1:imageCount
    croppedImages(:, :, 1, i) = cropImageRandom(squeeze(images(:, :, 1, i)), cx, cy);
end
end
