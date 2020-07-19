# Defect Classification/Detection
This repo compare methods to defect classificaton and detecton. Reference [2] gives a general introduction for recent methods in this field. The dataset I used is [DAGM 2007](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html) . 


## Getting Started

### Prerequisites
* Keras 2.4.3
* Tensorflow 2.2.0
* Opencv for python

### DAGM 2007 Dataset
There are 6 different data sets and each simulated using a different texture and defect model. Each data set has training set which is under Train folder and testing set which is under Test folder. A defect image and its mask image are shown below.
![](defect_mask.png) 

## Methods
1.Convolutional Neural Network based Classifier

a) [MobileNetV2](https://github.com/cvipdnn/defect_detection/tree/master/cnn/mobilenetv2)

b) [SimpleCNN](https://github.com/cvipdnn/defect_detection/tree/master/cnn/simplecnn) 


Method | a) | b) 
--- | --- | ---
Accuracy(testing set) | 99.971% | 85.913%
Multiplication FLOPs | 28944.9G | 63.2G

## Code Structure
1. cnn: Convolutional Neural Network based Classifier
2. utils: a tool used to analyze the performance of neural network, like multiplication FLOPs. 



## References
[1] https://www.kaggle.com/c/1056lab-defect-detection/data
[2] https://github.com/XiaoJiNu/surface-defect-detection
[3] https://conferences.mpi-inf.mpg.de/dagm/2007/

