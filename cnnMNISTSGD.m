[images, labels, net] = cnnMNISTInit('mnist', true);
images = images(:, :, 1);
res = vl_simplenn(net, images);
