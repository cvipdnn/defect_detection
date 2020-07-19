# Defect Detection/Classification
This repo compare several methods to defect classificaton and detecton. The dataset I used is [DAGM 2007](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html) . 


## Getting Started

### Prerequisites
* Keras 2.4.3
* Tensorflow 2.2.0
* Opencv for python


## Methods
There are 6 different data sets and each simulated using a different texture and defect model. Each data set has training set which is under Train folder and testing set which is under Test folder. 

1.Convolutional Neural Network based Classifier

a) [MobileNetV2](https://github.com/cvipdnn/defect_detection/tree/master/cnn/mobilenetv2)

b) [SimpleCNN](https://github.com/cvipdnn/defect_detection/tree/master/cnn/simplecnn) 


Method | a) | b) 
--- | --- | ---
Accuracy(testing set) | 99.971% | 85.913%




## References
* https://www.kaggle.com/c/1056lab-defect-detection/data
* https://github.com/XiaoJiNu/surface-defect-detection
* https://conferences.mpi-inf.mpg.de/dagm/2007/

