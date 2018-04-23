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
    trainedNet.test = testData;
    


end

function exportWeightsAndBiases(filename, network)
%    save(filename, [network.layers(2).Weights, network.layers(2).Bias, ...
%         network.layers(4).Weights, network.layers(4).Bias, ...
%         network.layers(7).Weights, network.layers(7).Bias, ...
%         network.layers(9).Weights, network.layers(9).Bias, ...
%         network.layers(12).Weights, network.layers(12).Bias, ...
%         network.layers(14).Weights, network.layers(14).Bias]);

end