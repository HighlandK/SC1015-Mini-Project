# Handwritten Digit Classification with MNIST Dataset

## Overview

This is a Mini-Project for SC1015 - Introduction to Data Science and Artificial Intelligence.

In this project, we leverage various machine learning models with the aim of creating a model which can recognise handwritten digits accurately. From the rudimentary Decision Tree Model, we progress towards the Logistic Regression, K-Nearest Neighbours and Convoluted Neural Network models, increasing in accuracy and sophistication each time. This project not only documents our results, but also our growth as data science students. 

## Problem Definition

- How might we leverage machine learning to automate the data entry process for cheques?
- Which model might be best to do so?

## Dataset

The MNIST dataset is a common dataset used for training computer vision models. Found on Kaggle, it is graciously contributed by Kaggler Dariel Dato-on. It comprises 70,000 images of handwritten digits (60,000 for training and 10,000 for testing - all independent). Each image is a 28x28 pixel grayscale picture flattened into a 784-dimensional vector. Labels indicating the actual digit (0-9) accompany each image. Note that git-lfs library was used to import the datasets as the train dataset was over the limit of 100MB. Altogether, the dataset contains 70,000 rows and 785 columns. For the best experience, please view the source code in the following order: 

- [Exploratory Analysis and Data Visualisation](https://github.com/Unknown-Blaze/SC1015-Mini-Project/blob/main/Exploratory%20Data%20Analysis%20and%20Data%20Visualisation.ipynb)
- [Decision Tree](https://github.com/Unknown-Blaze/SC1015-Mini-Project/blob/main/decision_tree.ipynb)
- [Logistic Regression](https://github.com/Unknown-Blaze/SC1015-Mini-Project/blob/main/Logistic%20Regression.ipynb)
- [K-Nearest Neighbours](https://github.com/Unknown-Blaze/SC1015-Mini-Project/blob/main/K%20Nearest%20Neighbours.ipynb)
- [Convoluted Neural Networks (CNN)](https://github.com/Unknown-Blaze/SC1015-Mini-Project/blob/main/cnn_code.ipynb)

## Exploratory Data Analysis and Visualization

The project begins with exploratory data analysis, where we explore the key characteristics of the dataset. These include: 
- Size of Dataset (Test and Train)
- Frequency of digit occurrences in dataset
- Ink Density (Proxied using pixel intensity)
- Frequency of pixel use

To do so, we have utilised the following data visualisation tools:
- Heatmap
- Box Plot
- Bar Chart

We have also done the following to prepare our data:
- [Data Normalisation] - Normalising pixel values
- [Data Grouping] - Grouping the data according to their digit labels
- [Data Reshaping] - One-hot-encoding

### Decision Tree

The first model we have implemented is the Decision Tree Classifier, existing in the sklearn library.  After multiple tries exploring different tree depth, we have determined 18 as the optimal depth of the tree - after which, the prediction accuracy of the tree decreases. 

On training the model, we visualise the decision process of the Decision Tree Classifier using a tree diagram.

Under the Decision Tree Classifier, we find that the maximum total accuracy attained is 88.29%.

### Logistic Regression

The second model we have implemented (out of syllabus) is the Logistic Regression model, also existing in the sklearn library. We conduct multinomial logistic regression, because we have more than two unordered types for our dependent variable. 

Under the Logistic Regression model, we find that the maximum total accuracy attained is 92.07%.


### K-Nearest Neighbours (KNN)

The third model we have implemented (out of syllabus) is the K-Nearest Neighbours model, existing in the sklearn library. In this model, we have set the number of neighbours as 5.

Under the K-Nearest Neighbours model, we find that the maximum total accuracy attained is 96.88%.

### Convoluted Neural Networks (CNN)

The fourth model we have implemented (out of syllabus) is the Convoluted Neural Network, which we have trained using stochastic gradient descent. Various numbers of nodes in the hidden layer (32, 64, 128, 256, 512, 1024) were implemented and evaluated. Each time we have plotted side-by-side graphs to visualise the improvements in the cross entropy loss as well as improvements in model prediction accuracy.

On evaluating the model, we have found that the maximum attained total accuracy for the Convoluted Neural Network is 97.92%, with 1024 nodes in the hidden layer. Note that running this code will always produce a different value, though with samll variations (97.90 ± 0.10)


## Results

- The prediction accuracies of the models rank in the following order (highest to lowest):
    - Convoluted Neural Network (CNN)
    - K-Nearest Neighbours (KNN)
    - Logistic Regression
    - Decision Tree
- Most of the models have consistent trouble prediction the digit '8' accurately, while the digit '1' enjoys the highest prediction accuracy. This could be due to the similarities of 8 with other numbers (e.g 6, 9 etc.) as compared to 1 with a more distinctive look.
- The models have a threshold, after which, their prediction accuracies do not increase any further. This is true for all the models used in this project.  

## Conclusion

From the results of the project, full automation in cheque data-entry is still not possible, but digit recognition can aid in performing the primary data entry for cheques, with bank staff assisting to confirm the information. 

## What did we learn from this project?

- K-Nearest Neighbours from sklearn
- Logistic Regression from sklearn
- Convoluted Neural Network (CNN)
- Normalising values

## Contributions

- Ganesh Rudra Prasadh (@Unknown-Blaze) - Convoluted Neural Network (CNN) and Github organisation
- Law Yu Chen (@y4y4y4y4y4) - Decision Tree, Logistic Regression, K-Nearest Neighbours (KNN) analysis
- Kow Zi Ting (@HighlandK) - Exploratory Data Analysis and Visualisation, Complete video editing and overall compilation

## Repository Structure

- **Code**: Separate Jupyter Notebooks for each model, affixed with comments explaining the process.
- **Data**: `mnist_train.csv` and `mnist_test.csv` files containing the dataset.
- **Images**: Sample images from the dataset and generated plots.

## References

- https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data
- https://medium.com/@AMustafa4983/handwritten-digit-recognition-a-beginners-guide-638e0995c826
- https://towardsdatascience.com/understanding-logistic-regression-9b02c2aec102
- https://medium.com/@AMustafa4983/handwritten-digit-recognition-a-beginners-guide-638e0995c826
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

## Quick Start Guide

To dive into the project:

1. Clone the repository.
2. Execute the notebooks to replicate our findings or experiment with different model configurations!
3. We hope you have a good time with the repository! :)
