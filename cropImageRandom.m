function croppedImage = cropImageRandom(image, cx, cy)
dy = size(image, 1) - cy + 1;
dx = size(image, 2) - cx + 1;
randInRange = @(a, b) round((b - a) .* rand(1) + a);
ystart = randInRange(1, dy);
xstart = randInRange(1, dx);
croppedImage = image(ystart:(ystart + cy - 1), xstart:(xstart + cx - 1));
end
