# ship-identification
Convolutional Neural Network written in MATLAB for classifying satellite images based on whether they contain a ship or not

### Dataset
This network was trained on data from this Kaggle dataset: https://www.kaggle.com/rhammell/ships-in-satellite-imagery

### Installation
With Matlab and the deep learning toolbox, this repository should be able to be cloned and opened ready to go. 
The dataset will need to be downloaded from the link above to train the model 

### Training Tips
One of the things I realized when training was that the dataset is not split 50/50. 
This dataset mostly has images with no ships in it. To effectively train your model, you will need to make the training set
have 50/50 representation while still keeping some other pictures aside for a good test set.

### Setup
#### Training Set
* 1000 images
* 500 contain ships
* 500 contain no ships

#### Testing Set
* The remaining 1800 images
* 200 contain ships
* 1600 contain no ships

### Results
The network was able to achieve 94.89% accuracy on the testing set after training (1708 out of 1800 correct).
