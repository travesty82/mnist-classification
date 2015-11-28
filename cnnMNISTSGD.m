[images, labels, net] = cnnMNISTInit('mnist', true);
images = images(:, :, 1:100);
res = vl_simplenn(net, images);
