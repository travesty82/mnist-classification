% SVK-KM must be in ./SVM-KM
% mnist data must be in ../mnist

setup

[Xtr, ytr, ~, ~] = loadMNIST('feat');

multisvm = multisvmtrain(Xtr(1:10000,:), ytr(1:10000));

save('multisvm.mat','-struct','multisvm');
