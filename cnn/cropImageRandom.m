function croppedImage = cropImageRandom(image, cx, cy)
% CROPIMAGERANDOM - Crops a random window out of an image.
%
% INPUTS
%   image - The image to crop (M x N)
%   cx - The width of the crop window
%   cy - The height of the crop window
%
% OUTPUTS
%   croppedImage - The cropped image (cx x cy)
%
dy = size(image, 1) - cy + 1;
dx = size(image, 2) - cx + 1;
randInRange = @(a, b) round((b - a) .* rand(1) + a);
ystart = randInRange(1, dy);
xstart = randInRange(1, dx);
croppedImage = image(ystart:(ystart + cy - 1), xstart:(xstart + cx - 1));
end
