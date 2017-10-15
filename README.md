# Self-Driving Cars Nanodegree
## Project 2: Traffic Sign Classifier 

* `Traffic_Sign_Classifier.ipynb` - Jupyter Notebook file with implementation of CNN using TensorFlow.
* `writeup_template.md` - project report
* `test_images/` - internet images for NN testing
* `report_images/` - images generated in `Traffic_Sign_Classifier.ipynb` for the report

### CNN Network Architecture

#### Layer 1: Convolutional. 
* Input: 32x32x3
* Filter: 5x5x18, valid padding
* Output: 28x28x18
* Activation: ReLU
* Max pooling: 2x2, same padding

#### Layer 2: Convolutional. 
* Input: 14x14x18
* Filter: 3x3x48, valid padding
* Output: 12x12x48
* Activation: ReLU
* Max pooling: 2x2, same padding

#### Layer 3: Convolutional. 
* Input: 6x6x48
* Filter: 3x3x96, valid padding
* Output: 4x4x96
* Activation: ReLU
* Max pooling: 2x2, same padding

#### Layer 4: Fully connected layer.
* Input: Layer 1 + Layer 2 + Layer 3 = 5640
* Output: 688
* Activation: ReLU
* Droput: 0.5

#### Layer 5: Fully connected layer.
* Input: 688
* Output: 86
* Activation: ReLU
* Droput: 0.5

#### Layer 6: Fully connected layer.
* Input: 86
* Output: 43Network architecture:

