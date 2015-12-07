function croppedImages = cropImageBatchRandom(images, cx, cy)
% CROPIMAGERANDOM - Crops a random window out of a batch of images.
%
% INPUTS
%   images - The images to crop (M x N x 1 x numberOfImages)
%   cx - The width of the crop window
%   cy - The height of the crop window
%
% OUTPUTS
%   croppedImages - The cropped images (cx x cy x 1 x numberOfImages)
%
imageCount = size(images, 4);
croppedImages = zeros(cx, cy, 1, imageCount);
for i = 1:imageCount
    croppedImages(:, :, 1, i) = cropImageRandom(squeeze(images(:, :, 1, i)), cx, cy);
end
end
