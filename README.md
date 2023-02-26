# image-classification-api
Image classification API using Tensorflow and Flask. The API takes a picture of a digit as an input and predicts the digit class.

### Features:
#### 1. Building and training a CNN model using Tensorflow and Keras libraries.
#### 2. Providing an API interface for deploying the trained model using Flask and testing in Flasgger web client.


## Files Information
### 1. image_recognition_train.py
#### The main file for constructing a CNN network and training it over MNIST data set. The trained model is saved for later to be used by the web API

### 2. image_recognition_test.py
#### Contains the decorator for flask API which performs the tasks for predicting the image class.

### 3. requirements.txt
#### The file contains the list of all the required libraries


## Running Project
#### 1. Run image_recognition_train.py file for training the model. The model will be saved as a .h5 file
#### 2. Run the following command in the terminal
####    python image_recognition_test.py 
#### 3. The project will start and a link will be displayed in the terminal.
#### 4. Click on the link. When the browser opens up, write '/apidocs' after hit enter to and open the URL and run flasgger interface.
#### e.g., if the URL generated is 'http://127.0.0.1:5000', then the final URL will be 'http://127.0.0.1:5000/apidocs'
