# Amazon Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using various text processing and machine learning techniques.

## Overview

The analysis aims to predict whether a product review is positive or negative based on its text content. Reviews with scores above 3 are considered positive, while those below or equal to 3 are labeled negative.

## Data Processing Pipeline

### 1. Text Preprocessing
- Lowercasing
- HTML tag removal
- Tokenization
- Punctuation removal
- Stop word removal
- Stemming

### 2. Text Vectorization Methods
- **Unigram**: Individual word features
- **Bigram**: Two consecutive word features
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Word2Vec**: Word embedding technique

### 3. Machine Learning Models
Two classification models were implemented:
- Logistic Regression
- Random Forest

## Results

### Model Performance
Both models achieved similar performance metrics:
- Accuracy: ~88% (using Unigram)
- Positive class:
  - Recall: 96%
  - F1 score: 0.93
- Negative class:
  - Recall: 63%
  - F1 score: 0.70

### Feature Importance Analysis
Top important features identified:
- Unigram: "great", "love", "best", "delici"
- Bigram: "wast money", "wo buy", "never buy", "highly recommend"

## Dependencies
- pandas
- numpy
- scikit-learn
- nltk
- BeautifulSoup
- gensim
- matplotlib

## Usage

1. Install required packages:

```bash
pip install pandas numpy scikit-learn nltk beautifulsoup4 gensim matplotlib
```
2. Load and preprocess data:
```python
import pandas as pd
df = pd.read_csv('archive/Reviews.csv')
```

3. Train models:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=5, random_state=42)
model.fit(X_train, y_train)
```
4. Save/Load model:
```python
import joblib
joblib.dump(model, 'random_forest_model.pkl')
model = joblib.load('random_forest_model.pkl')
```

## Key Findings

1. Unigram achieved the best performance with high accuracy (0.89) and AUC (0.93)
2. Logistic Regression slightly outperformed Random Forest in AUC scores
3. The models show better performance in identifying positive reviews compared to negative ones
4. Data imbalance (positive:negative = 3.56:1) might affect model performance

## Future Improvements
1. Address class imbalance
2. Experiment with more features for Bigram analysis
3. Implement deep learning models
4. Optimize hyperparameters