%% Citations
% I used the following links to generate some of these graphics:
% ---- https://www.mathworks.com/help/nnet/examples/visualize-activations-of-a-convolutional-neural-network.html

%% Get Pretty Figures fast with Feedforward mode only (No Training)
% Train the network once and get the pictures here so it is faster
% to generate figures.

function GetImages(network, testData)
    %getMontage(network, 380, testData);
    getTestExamples(network, testData);
end

%% Get Conv Progressions of an Image
function getMontage(network, image, testData)
    i = image;

    I = readimage(testData, i);
    act1 = activations(network, I, 'conv1', 'OutputAs', 'channels');
    act2 = activations(network, I, 'conv2', 'OutputAs', 'channels');
    act3 = activations(network, I, 'conv3', 'OutputAs', 'channels');

    sz1 = size(act1);
    sz2 = size(act2);
    sz3 = size(act3);

    act1 = reshape(act1,[sz1(1) sz1(2) 1 sz1(3)]);
    act2 = reshape(act2,[sz2(1) sz2(2) 1 sz2(3)]);
    act3 = reshape(act3,[sz3(1) sz3(2) 1 sz3(3)]);
    
    figure
    montage(act1,'Size',[8 8])
    figure
    montage(act2,'Size',[8 8])
    figure
    montage(act3,'Size',[8 16])
end

%% Output Example Images with Labels
function getTestExamples(network, testData)
    shipPredictions = classify(network, testData);
    correctLabels = testData.Labels;    

    figure
    p = 1:10:400;
    for i = 1:numel(p)
        subplot(5, 8, i);
        
        I = readimage(testData, p(i));
        
        label = strcat(char(correctLabels(p(i))), " vs. ", char(shipPredictions(p(i))));
        
        imshow(I)
        title(label)
    end
end