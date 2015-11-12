addpath('third-party/mnistHelper');

train_images = loadMNISTImages('mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

x = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
t = [1 1 1 0];
train_perceptron(x, t, 0.1, @unipolar_tfn)
