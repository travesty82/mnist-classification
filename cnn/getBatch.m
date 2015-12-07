function [imb, lb] = getBatch(images, labels, batchSize, batchNumber)
% GETBATCH - Gets a batch of images and labels.
%
% INPUTS
%   images - The source images (M x N x 1 x numImages)
%   labels - The source labels (1 x numImages)
%   batchSize - The number of images per batch
%   batchNumber - The batch index
%
% OUTPUTS
%   imb - The images in the batch (M x N x 1 x batchSize)
%   lb - The labels in the batch (1 x batchSize)
%
batchStart = (batchSize * (batchNumber - 1)) + 1;
batchEnd = min(batchStart + batchSize - 1, size(labels, 2));
imb = images(:, :, :, batchStart:batchEnd);
lb = labels(:, batchStart:batchEnd);
end
