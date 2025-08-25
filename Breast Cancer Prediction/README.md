# Breast Cancer Prediction using Neural Networks

## Overview
This project leverages machine learning, specifically neural networks, to predict breast cancer malignancy using clinical diagnostic data. The goal is to distinguish between benign (non-cancerous) and malignant (cancerous) tumors with high accuracy, supporting early detection and better patient outcomes.

**Check out the kaggle notebook over here:** [Kaggle Notebook Link](https://www.kaggle.com/code/bengj10/breast-cancer-prediction-with-ann-explained)

## Motivation
Breast cancer is one of the most common cancers worldwide. Early detection is critical for effective treatment and improved survival rates. Traditional diagnostic methods, while effective, can be time-consuming, costly, and prone to human error. Machine learning models can help automate and enhance the accuracy of these predictions, assisting healthcare professionals in making data-driven decisions.

## Problem Statement
- **Task:** Binary classification — predict whether a tumor is malignant (1) or benign (0) from numeric features computed from digitized images of fine needle aspirate (FNA) of a breast mass.
- **Approach:** Use a neural network (NN) for flexible, non-linear modeling and compatibility with gradient-based explainability.

## Dataset
- **Source:** `dataset/breast-cancer.csv`
- **Features:** Numeric attributes derived from FNA images (e.g., radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Diagnosis (`M` = Malignant, `B` = Benign)


## Workflow
1. **Data Loading & Cleaning**
   - Load the dataset and drop unnecessary columns (e.g., `id`).
   - Check for and handle missing values (none in this dataset).
2. **Exploratory Data Analysis (EDA)**
   - Visualize class distribution and feature statistics.
   - Use violin and swarm plots to understand feature distributions by class.
   - Analyze feature correlations.
3. **Preprocessing**
   - Encode the target variable (`diagnosis`) to numeric labels.
   - Split data into features (`X`) and target (`Y`).
   - Standardize features using `StandardScaler`.
4. **Model Building**
   - Construct a neural network using TensorFlow/Keras.
   - Architecture: Input layer (flatten), one hidden layer (ReLU), output layer (softmax/sigmoid for binary classification).
   - Compile with appropriate loss and optimizer.
5. **Training & Evaluation**
   - Train the model with validation split.
   - Plot training/validation accuracy and loss.
   - Evaluate model performance on the test set.
6. **Prediction System**
   - Predict on new/unseen data points.
   - Output whether the tumor is likely benign or malignant.

## Key Results
- The neural network achieves high accuracy in distinguishing between benign and malignant tumors.
- Visualizations reveal which features are most informative for prediction.
- The model can be used as a predictive system for new patient data.

## License
This project is licensed under the terms of the LICENSE file in this repository.

## Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- Inspired by the need for accurate, accessible, and interpretable AI in healthcare.
