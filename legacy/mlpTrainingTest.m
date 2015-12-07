x = [0 0; 0 1; 1 0; 1 1];
t = [0; 1; 1; 0];
n = size(x, 1);

hiddenNeurons = 4;
learningRate = 0.75;
momentum = 0.001;
numIter = 1000;

[mlp, mse] = mlpTrain([hiddenNeurons], x, t, learningRate, numIter, 'logsig', 'momentum', alpha);
y = zeros(1, n);
for i = 1:n
    y(i) = mlpPredict(mlp, x(i, :));
end

display(y, 'Predicted Output');
figure, plot(1:numIter, mse);
ylabel('Mean Squared Error');
xlabel('Epoch');
title('MSE vs. Epoch');
