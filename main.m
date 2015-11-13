addpath('third-party/mnistHelper');

train_images = loadMNISTImages('mnist/train-images-idx3-ubyte');
train_labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

x = [1 0 0; 1 0 1; 1 1 0; 1 1 1];
t = [1 1 1 0];
p = perceptron_train(x, t, 0.1, @unipolar_tfn);
perceptron_predict(p, [1 0 0]);

x = [0 0; 0 1; 1 0; 1 1];
t = [1; 1; 1; 0];
mlp = mlp_train([3 2], x, t, 0.1, 'logsig', 'threshold', 0.4);
