%% Basic structure of the CNN
% The architecture is going to be based on this formula:
% INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
% Where 0 <= N <= 3, M >= 0, 0 <= K < 3
% I am using Max Pooling currently
% Choosing N = 2, M = 2, K = 1

%% Build the Network
function network = ShipConvNet()

%% Create the Input Layer
% The input is an 80x80 RGB image, so we need a size of [80 80 3]
inputLayer = imageInputLayer([80 80 3], 'Name', 'input');

%% First Convolutional Layer
% Spatial Extent: 3x3
% Stride: 1
% Zero Padding: 1
% Number of filters: 16
convLayer1 = convolution2dLayer(3, 64, 'Stride', 1, 'Padding', 1, 'Name', 'conv1');

%% First ReLU Layer
reluLayer1 = reluLayer();

%% Second Convolutional Layer
% Spatial Extent: 3x3
% Stride: 1
% Zero Padding: 1
% Number of Filters: 16
convLayer2 = convolution2dLayer(3, 64, 'Stride', 1, 'Padding', 1, 'Name', 'conv2');

%% Second ReLU Layer
reluLayer2 = reluLayer();

%% First Max Pooling Layer
% Receptive Field: 2x2
% Stride: 2
poolLayer1 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1');

%% Third Convolutional Layer
% Spatial Extent: 3x3
% Stride: 1
% Zero Padding: 1
% Number of Filters: 8
convLayer3 = convolution2dLayer(3, 128, 'Stride', 1, 'Padding', 1, 'Name', 'conv3');

%% Third ReLU Layer
reluLayer3 = reluLayer();

%% Fourth Convolutional Layer
% Spatial Extent: 3x3
% Stride: 1
% Zero Padding: 1
% Number of Filters: 8
convLayer4 = convolution2dLayer(3, 128, 'Stride', 1, 'Padding', 1, 'Name', 'conv4');

%% Fourth ReLU Layer
reluLayer4 = reluLayer();

%% Second Max Pooling Layer
% Receptive Field: 2x2
% Stride: 2
poolLayer2 = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2');

%% First Fully Connected Layer
% Input Dimensions: 20x20x32
% Output Dimensions: 1x1x20
fcLayer1 = fullyConnectedLayer(64, 'Name', 'full1');

%% Fifth ReLU Layer
reluLayer5 = reluLayer();

%% Second Fully Connected Layer
% Input Dimensions: 1x20x20
% Output Dimensions: 1x1x2
fcLayer2 = fullyConnectedLayer(2, 'Name', 'full2');

%% Softmax Layer
softLayer = softmaxLayer();

%% Classification Layer (Output Layer)
outputLayer = classificationLayer('Name', 'output');

%% Create Layers Vector
network.layers = [inputLayer, convLayer1, reluLayer1, convLayer2, ...
    reluLayer2, poolLayer1, convLayer3, reluLayer3, ...
    reluLayer4, poolLayer2, fcLayer1, reluLayer5, fcLayer2, softLayer, outputLayer];

%% Add Local Function Access
network.trainingInit = @trainingInit;
network.savedInit = @savedInit;
end

%% Network Initialization (Training)
function layers = trainingInit()
    % Convolutional Layer 1
    % Initialize weights and biases
    layers(2).Weights = randn([3 3 3 64]) * 0.001;
    layers(2).Bias = (randn([1 1 64]) * 0.001) + 0.333;

    % Convolutional Layer 2
    % Initialize weights and biases
    layers(4).Weights = randn([3 3 3 64]) * 0.001;
    layers(4).Bias = (randn([1 1 64]) * 0.001) + 0.333;
    
    % Convolutional Layer 3
    % Initialize weights and biases
    layers(7).Weights = randn([3 3 3 128]) * 0.001;
    layers(7).Bias = (randn([1 1 128]) * 0.001) + 0.333;
    
    % Convolutional Layer 4
    % Initialize weights and biases
    layers(9).Weights = randn([3 3 3 128]) * 0.001;
    layers(9).Bias = (randn([1 1 128]) * 0.001) + 0.333;
    
    % Fully Connected Layer 1
    % Initialize weights and biases
    layers(12).Weights = randn([64 3200]) * 0.001;
    layers(12).Bias = (randn([64 1]) * 0.001) + 0.5;
    
    % Fully Connected Layer 2
    % Initialize weights and biases
    layers(14).Weights = randn([2 64]) * 0.001;
    layers(14).Bias = (randn([2 1]) * 0.001) + 0.5;  
end


%% Network Initialization from Trained Values
function layers = savedInit(filepath)
    
end

