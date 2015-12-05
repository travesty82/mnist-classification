function net = cnnMNISTInit(id, options)
    if id == 1
        net = cnnMNISTInitConvnetjs(options);
    elseif id == 2
        net = cnnMNISTInitTensorFlow(options);
    elseif id == 3
        net = cnnMNISTInitMatconvnet(options);
    end
end
