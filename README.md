# Complete Deep Learning Projects 

This repository contains a collection of **deep learning projects** focusing on real-world problem solving using various neural network architectures, including **Convolutional Neural Networks (CNNs)**, **Artificial Neural Networks (ANNs)**, and **Recurrent Neural Networks (RNNs)** with LSTM layers and more. The projects demonstrate practical applications of deep learning in image classification, tabular data analysis, and natural language processing.

---

## Projects

### [1. Plant Disease Detection using CNN](./Plant%20Disease%20Detection%20using%20CNN)

**Objective:**  
Classify plant leaf images into healthy or diseased categories.

**Architecture:**  
- **Convolutional Neural Network (CNN)** with multiple convolutional layers followed by max-pooling.  
- Fully connected dense layers for classification.  
- Softmax activation in the output layer for multi-class classification.

**Methodology:**  
1. Image preprocessing: resizing, normalization, and data augmentation (rotation, flipping, zoom).  
2. Model building: sequential CNN architecture with ReLU activation.  
3. Training: cross-entropy loss, Adam optimizer, and early stopping based on validation loss.  
4. Evaluation: accuracy, confusion matrix, precision, recall, and F1-score.

**Results:**  
- Achieved approximately 95% accuracy on validation and test datasets.  
- Model effectively distinguishes between multiple disease classes with minimal misclassification.

---

### [2. Customer Churn Prediction using ANN](./Customer%20Churn%20using%20ANN)

**Objective:**  
Predict whether a customer is likely to churn using structured tabular data.

**Architecture:**  
- **Artificial Neural Network (ANN)** with multiple fully connected layers.  
- ReLU activation for hidden layers and Sigmoid activation for output layer.  
- Binary classification for churn (Yes/No).

**Methodology:**  
1. Data preprocessing: encoding categorical variables, handling missing values, feature scaling.  
2. Model training: binary cross-entropy loss, Adam optimizer, and batch gradient descent.  
3. Evaluation: accuracy, ROC-AUC score, confusion matrix.

**Results:**  
- Model achieved 86–87% accuracy on validation data.  
- Feature importance analysis indicates key predictors of churn, enabling actionable business insights.

---

### [3. Breast Cancer Classification using ANN](./Breast%20Cancer%20Prediction)

**Objective:**  
Classify breast tumors as **malignant** or **benign** based on clinical features.

**Architecture:**  
- Fully connected ANN with input, hidden, and output layers.  
- ReLU activation for hidden layers; Sigmoid activation for binary classification.

**Methodology:**  
1. Data preprocessing: standardization (z-score normalization).  
2. Model training: sparse categorical cross-entropy loss, Adam optimizer.  
3. Evaluation: accuracy, ROC-AUC, and confusion matrix.

**Results:**  
- Achieved approximately 97% accuracy on test data.  
- High precision and recall for malignant class, critical for medical diagnostics.

---

### [4. Movie Sentiment Analysis using LSTM (RNN)](./Movie%20Sentiment%20Analysis%20using%20LSTM)

**Objective:**  
Classify IMDB movie reviews as **positive** or **negative** based on textual data.

**Dataset:**  
IMDB Review Dataset consisting of 50,000 labeled reviews with an equal distribution of positive and negative sentiments.

**Architecture:**  
- **Embedding Layer:** Converts integer-encoded words into dense vector representations (word embeddings).  
- **LSTM Layer:** Captures long-term dependencies and contextual relationships in sequences.  
- **Dense Output Layer:** Single neuron with Sigmoid activation for binary classification.

**Methodology:**  
1. Data preprocessing:  
   - Text cleaning, removing HTML tags, punctuation, numbers, and stopwords.  
   - Tokenization and integer encoding of words.  
   - Sequence padding to a uniform length.  
2. Model training:  
   - Binary cross-entropy loss, Adam optimizer, and early stopping based on validation loss.  
   - Batch size: 32; Epochs: 10–20.  
3. Evaluation: accuracy on test set and visualization of training/validation loss and accuracy.

**Results:**  
- Simple RNN achieved ~86% accuracy.  
- LSTM improved performance to 90%+ accuracy due to better handling of long-term dependencies.  
- Confusion matrices confirm strong prediction capability on both positive and negative reviews.

---

## Technologies and Libraries

- **Programming Language:** Python 3.12+  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Utilities:** Scikit-learn  

---

## Future Improvements

- Experiment with pre-trained embeddings (GloVe, Word2Vec, BERT) for NLP tasks.

- Implement Bidirectional LSTM or GRU for improved sequential modeling.

- Deploy models using Flask/Django or FastAPI for production.

- Perform hyperparameter optimization (Grid Search, Bayesian Optimization).

- Incorporate regularization techniques (Dropout, L2) to enhance generalization.

---

## License

This repository is licensed under the MIT License.

---

## Author

**Ben Gregory John**  
B.Tech CSE Student | Specializing in Machine Learning, AI, and MLOps  

[LinkedIn](https://www.linkedin.com/bengj10) | [GitHub](https://github.com/BenGJ10) 
