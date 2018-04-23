%% Citations
% Some code here was taken from the following MATLAB Examples:
% ---- Transfer Learning and Fine-Tuning of Convolutional Neural Networks

function trainedNet = ShipNetworkTrainer(isNew)
    %% Pre-Training File Location
    trainedNet.preTrainingFilePath = "ShipConvNet_WeightsAndBiases.mat";
    
    %% Get the CNN Architecture
    net = ShipConvNet();
    
    %% Initialize the Network Weights
    if nargin == 1
        if isNew
            net.trainingInit();
        else
            net.savedInit();
        end
    else
        net.trainingInit();
    end
    
    %% Load the Images of Ships
    shipDatasetPath = fullfile('..','dataset');

    shipData = imageDatastore(shipDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

    [trainingData,testData] = splitEachLabel(shipData, 500, 200);
    
    %% Train the Network
    % Set training options
    trainOptions = trainingOptions('sgdm', 'MiniBatchSize', 100, ...
        'MaxEpochs', 20, 'Momentum', 0.925, 'InitialLearnRate', 0.002,...
        'ExecutionEnvironment', 'gpu');
    
    %Actually do the training
    network = trainNetwork(trainingData, net.layers, trainOptions);
    
    %Test on the test set and print accuracy
    shipPredictions = classify(network, testData);
    correctLabels = testData.Labels;
    
    testAccuracy = sum(shipPredictions == correctLabels) / numel(correctLabels)
    
    %% Export the Weights and Biases
    exportWeightsAndBiases(trainedNet.preTrainingFilePath, network);
    
    %% Return the Trained Network
    trainedNet.accuracy = testAccuracy;
    trainedNet.network = network;
    
    %% Output Example Images
    figure
    p = 1:20:400;
%     for i = 1:numel(p)
%         subplot(4, 5, i);
%         
%         I = readimage(testData, p(i));
%         
%         label = strcat(char(correctLabels(p(i))), " vs. ", char(shipPredictions(p(i))));
%         
%         imshow(I)
%         title(label)
%     end
    
    for i = 1:numel(p)
        figure
        I = readimage(testData, p(i));
        act1 = activations(network, I, 'conv1', 'OutputAs', 'channels');
        act2 = activations(network, I, 'conv2', 'OutputAs', 'channels');
        act3 = activations(network, I, 'conv3', 'OutputAs', 'channels');
        
        sz1 = size(act1);
        sz2 = size(act2);
        sz3 = size(act3);
        
        act1 = reshape(act1,[sz1(1) sz1(2) 1 sz1(3)]);
        act2 = reshape(act2,[sz2(1) sz2(2) 1 sz2(3)]);
        act3 = reshape(act3,[sz3(1) sz3(2) 1 sz3(3)]);
        
        allActs = cat(3, act1, act2, act3);
        
        montage(allActs,'Size',[24 32])
    end
end

function exportWeightsAndBiases(filename, network)
%    save(filename, [network.layers(2).Weights, network.layers(2).Bias, ...
%         network.layers(4).Weights, network.layers(4).Bias, ...
%         network.layers(7).Weights, network.layers(7).Bias, ...
%         network.layers(9).Weights, network.layers(9).Bias, ...
%         network.layers(12).Weights, network.layers(12).Bias, ...
%         network.layers(14).Weights, network.layers(14).Bias]);

end