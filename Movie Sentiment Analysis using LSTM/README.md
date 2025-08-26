# IMDB Movie Review Sentiment Analysis

This project performs **sentiment analysis** on the **IMDB movie reviews dataset** using **Recurrent Neural Networks (RNN)** and explores how **Long Short-Term Memory (LSTM)** networks improve performance for sequence-based text tasks.

---

## Problem Statement
Given a movie review (text), predict whether the review is **positive** or **negative**.  
This is a **binary classification** problem in Natural Language Processing (NLP).

**Dataset Link**: [IMDB Movie Reviews](https://kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Concepts Used

### 1. Word Embedding & Feature Representation
- Raw text cannot be fed directly into neural networks.  
- We convert words into **numeric vectors** using **word embeddings**.  
- Embeddings map words into a continuous vector space where **semantic relationships** are preserved.  
  - Example: *king* → [0.25, -0.63, 0.41, ...]  
  - Words with similar meaning have embeddings close to each other.  

### 2. Recurrent Neural Network (RNN)

- RNNs are designed for **sequential** data (like text).

- At each step, the RNN cell takes the current word embedding and information from previous words.

- This helps capture dependencies in text, e.g.

    - "The movie was not good" → "not" changes the sentiment drastically.

- However:

    - **Vanishing gradients problem** → long-range dependencies are hard to learn.

    - Example: RNN may forget the word not when processing later words in a long review.


### 3. Long Short-Term Memory (LSTM)

- LSTM is an advanced RNN variant.

- Uses gates (input, forget, output) to control memory flow.

- Benefits:

    - Remembers important context for long sequences.

    - Ignores irrelevant words.

    - Solves vanishing gradient issue better than vanilla RNN.

For sentiment analysis:

- LSTM can remember that "not" (appearing at the start) should still affect "good" (appearing later).

- Hence, **LSTMs usually outperform simple RNNs**.


---

## Project Workflow

1. **Data Preparation**  
   - Load IMDB dataset (50,000 reviews).  
   - Tokenize and pad sequences.  
   - Train/Test split.

2. **Model Building**  
   - RNN model with an Embedding layer.  
   - LSTM model for better long-term sequence handling.  
   - Early stopping to avoid overfitting.

3. **Training**  
   - Optimizer: Adam  
   - Loss: Binary Crossentropy  
   - Metrics: Accuracy

---

## Model Architecture

**For Simple RNN**:
```python
model = Sequential()
model.add(Embedding(10000, 128))
model.add(SimpleRNN(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
```

**For LSTM (improved version)**:
```python
model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```

---

## Results

- **Simple RNN** achieved decent accuracy but struggled with long reviews.  
- **LSTM** improved performance by handling long-term dependencies better.

**RNN Performance on Test Set:**  
- **Accuracy:** 0.86  
- **Loss:** 0.32

**LSTM Performance on Test Set:**  
- **Accuracy:** 0.88  
- **Loss:** 0.28

--- 

## References
- [IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- [Understanding RNNs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
- [Keras Embedding Layer Docs](https://keras.io/api/layers/core_layers/embedding/)  

---