# Plant Disease Detection using CNN

This project implements a Convolutional Neural Network (CNN) model for detecting plant diseases based on leaf images. The approach is inspired by the research paper, [Plant Disease Detection](https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf_icacc2022_03049.pdf). The system leverages deep learning techniques to assist in early identification of crop diseases, thereby supporting farmers in mitigating potential agricultural losses.

**Check out the kaggle notebook over here:** [Kaggle Notebook Link](https://www.kaggle.com/code/bengj10/plant-disease-prediction-cnn-with-explanation)

---

## Problem Statement
Plant diseases caused by pests, bacteria, or environmental conditions significantly reduce agricultural productivity. Traditional methods of identifying plant diseases are time-consuming and often require expert knowledge. This project aims to automate disease detection using image classification techniques, enabling faster, scalable, and more accurate diagnosis.

---

## Project Objectives
- Develop a CNN-based image classification model to detect plant diseases.

- Train the model on leaf images of healthy and diseased plants.

- Evaluate the model’s performance using standard metrics (accuracy, loss).

- Provide predictions for unseen plant leaf images.

---

## Dataset
- **Source**: Publicly available plant disease image datasets (e.g., PlantVillage).  
- **Content**: Thousands of leaf images across multiple plant species and disease categories.  
- **Preprocessing**:  
  - Image resizing to a fixed shape (e.g., 224×224).  
  - Normalization (scaling pixel values between 0 and 1).  
  - Train-test split to ensure unbiased evaluation.

---

## Methodology

### 1. Data Preprocessing
- Loaded images using `PIL` and converted them into NumPy arrays.
- Resized all images to uniform dimensions.
- Normalized pixel values to improve training stability.
- One-hot encoded labels for multi-class classification.

### 2. Model Architecture (CNN)
The CNN model was built using TensorFlow/Keras with the following layers:

1. **Input Layer**  
   - Shape: (224, 224, 3)

2. **Convolutional Layers**  
   - Multiple convolutional layers with filters (32, 64, 128) and kernel size (3×3).  
   - Activation function: ReLU  
   - Purpose: Extract spatial features from input images.

3. **Pooling Layers**  
   - MaxPooling layers (2×2) applied after convolutional layers.  
   - Purpose: Reduce dimensionality while retaining important features.

4. **Dropout Layers**  
   - Dropout applied (rate: 0.25–0.5) to reduce overfitting.

5. **Flatten Layer**  
   - Converts 2D feature maps into a 1D feature vector.

6. **Fully Connected (Dense) Layers**  
   - Dense layers with ReLU activation for high-level reasoning.  
   - Final output Dense layer with **Softmax activation** for multi-class classification.

### 3. Model Compilation
- Optimizer: **Adam**  
- Loss function: **Categorical Crossentropy** (multi-class setup)  
- Metrics: **Accuracy**

### 4. Training
- Performed over multiple epochs with batch training.  
- Training and validation accuracy/loss tracked.  
- Early stopping and learning rate adjustments were optionally applied.

### 5. Evaluation
- Accuracy and loss curves analyzed to assess model convergence.  
- Evaluated using test dataset.  
- Classification performance measured by confusion matrix and accuracy score.

### 6. Prediction
- Implemented a utility function to:
  - Load and preprocess new images.
  - Predict disease class using the trained CNN.
  - Map output predictions to class labels.

---

## Results
- Achieved high training and validation accuracy on the dataset.  
- Demonstrated reliable disease classification capability across multiple plant categories.  
- Successfully predicted disease class for unseen leaf images.

---

## Tools and Frameworks
- **Programming Language**: Python  
- **Libraries**:  
  - TensorFlow/Keras (model building and training)  
  - NumPy, Pandas (data handling)  
  - Matplotlib, Seaborn (visualizations)  
  - PIL (image preprocessing)  

---

## Future Improvements
- Extend model to more plant species and disease classes.  
- Deploy as a web or mobile application for real-time farmer usage.  
- Optimize model performance with transfer learning using pre-trained architectures (e.g., ResNet, VGG, MobileNet).  
- Incorporate explainability techniques (Grad-CAM) to highlight diseased regions in leaves.

---

## References
- Original research paper: [Plant Disease Detection](https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf_icacc2022_03049.pdf)  
- TensorFlow/Keras official documentation: https://www.tensorflow.org/  
- PlantVillage dataset: https://plantvillage.psu.edu/

---
