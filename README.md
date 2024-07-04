# American-Sign-Language-Recognition

## Introduction
American Sign Language (ASL) is a vital mode of communication for individuals with hearing impairments. Advances in technology have led to the development of systems that bridge the communication gap between sign language users and the wider community. This unique and expressive language relies on handshapes, facial expressions, and body movements to convey meaning. 

## Process
In this project we aimed to enhance the prospects of American Sign Language by using Convolutional Neural Network (CNN). We acquired a dataset from Kaggle, preprocessed the data to eliminate irrelevant features, and trained a machine learning model on sign language gesture images. The trained model is capable of converting hand and finger movements captured through a webcam into text, thereby enabling seamless communication between ASL users and those unfamiliar with sign language.

## Dataset
We have provided the link to the dataset obtained from Kaggle right below. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z. No cases for 9=J or 25=Z have been included since these alphabets involve motion gestures that cannot be represented in static images. The training data consists of 27,455 datapoints and test data consists of 7172 datapoints. These are approximately half the size of the standard MNIST. Images are 28x28 pixel each with grayscale values between 0-255.

https://www.kaggle.com/datasets/datamunge/sign-language-mnist <br/><br/>

![ASL_gestures](https://github.com/Ipshita-Tandon/American-Sign-Language-Recognition/assets/120296010/afa1f73a-e0ff-4141-a0bf-a9a95278de1a)

## Libraries Used

### NumPy
1. Data Manipulation: Handling and processing the numerical data, especially the image pixel values which are represented as arrays.
2. Mathematical Operations: Performing various mathematical computations needed for preprocessing the images, such as normalization.

### Pandas
1. Data Loading: Reading the dataset from CSV files provided by Kaggle.
2. Data Cleaning: Filtering and preprocessing the dataset to remove any irrelevant or redundant features.
3. Data Analysis: Conducting exploratory data analysis (EDA) to understand the distribution and characteristics of the dataset.

### Scikit-learn
1. Data Splitting: Dividing the dataset into training and testing sets to evaluate the performance of the model.
2. Model Evaluation: Using metrics like accuracy and confusion matrix to assess the performance of the model.
3. Preprocessing: Techniques like scaling and normalization of data to improve the efficiency of model training.

### Matplotlib and Seaborn
1. Data Visualization: Visual representation of the dataset to plot and understand relationships between features.
2. Model Performance: Plotting the training and validation accuracy and loss to analyze the model's efficiency.
3. Seaborn: Helped in presenting more advanced, aethetically pleasing visuals.

### Keras
1. Model Building: Defining the architecture of the Convolutional Neural Network (CNN).
2. Model Training: Training the CNN on the preprocessed dataset.
3. Model Evaluation: Evaluating the performance of the CNN on the test dataset.

### CV2
1. Image Preprocessing: Resizing and normalization of images to prepare them for model classification.
2. Real-time Image Capture: Capturing images from the webcam to test the model's ability to recognize ASL gestures in real-time.


