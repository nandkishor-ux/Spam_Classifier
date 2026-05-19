# Spam Classifier — NLP Project

A machine learning project that classifies SMS messages as spam or ham using text preprocessing and Logistic Regression.
---
## Overview
This project was built as part of learning Natural Language Processing. It covers the full pipeline from raw text to a trained classifier — including tokenization, lemmatization, TF-IDF feature extraction, and model evaluation.
---
## Dataset

- **File:** `spam.csv`
- **Size:** 5575 messages
- **Source:** SMS Spam Collection Dataset (UCI ML Repository)
- **Class distribution:** ~87% ham, ~13% spam
- **Encoding:** latin-1

---

## Project Structure

```
natural_language_processing/
├── dataset/
│   └── spam.csv
├── spam_classifier_model/
│   └── spam_classifier.ipynb
└── README.md
```

---

## Pipeline

### 1. Data Loading and Cleaning
- Loaded CSV with `pandas` using `latin-1` encoding
- Dropped NaN rows from message column
- Filtered only ham and spam labeled rows
- Converted all messages to string to avoid type errors

### 2. Text Preprocessing
Each message goes through the following steps:

- **Lowercasing** — convert all text to lowercase
- **URL replacement** — replace links with the token `url`
- **Number replacement** — replace digits with the token `num`
- **Punctuation removal** — strip all special characters
- **Tokenization** — split sentence into individual words using NLTK
- **Stopword removal** — remove common words like "the", "is", "a"
- **Lemmatization** — reduce words to base form, "prizes" becomes "prize"

### 3. Feature Engineering
- Used `TfidfVectorizer` from scikit-learn
- Top 5000 features selected
- `sublinear_tf=True` to handle repeated words in spam messages
- Vectorizer fitted only on training data to avoid data leakage

### 4. Model Training and Testing
- Algorithm used: **Logistic Regression**
- 80/20 train test split with `stratify=y` to preserve class ratio
- Evaluated using precision, recall, F1 score and confusion matrix

---

## Libraries Used

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Install all at once:

```
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```
---

## NLTK Downloads

Run this once before starting:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```


## How to Run

**Step 1** — Install dependencies
```
pip install -r requirements.txt
```
**Step 2** — Open the notebook
```
jupyter notebook spam_classifier.ipynb
```
**Step 3** — Run all cells from top to bottom
---
## Results

```
              precision    recall    f1-score    support
ham              0.99       0.99       0.99        966
spam             0.95       0.96       0.96        149
accuracy                               0.98       1115
macro avg        0.97       0.97       0.97       1115
```
---
## Key Learnings

- **Data leakage** is the most common mistake — always fit the vectorizer on training data only, never on the full dataset
- **Accuracy is misleading** on imbalanced data — F1 score is the right metric here since spam is only 13% of the data
- **TF-IDF beats raw word counts** — down-weighting common words improves classification noticeably
- **Lemmatization reduces sparsity** — fewer unique tokens means a cleaner feature space
---
## What Can Be Improved

- Add bigrams using `ngram_range=(1,2)` for phrase-level signals like "free entry" and "click here"
- Try LinearSVC which often outperforms Logistic Regression on text
- Use cross-validation for more reliable evaluation
- Add POS-aware lemmatization for better verb handling
---
## Author

Built as part of NLP learning journey covering tokenization, lemmatization, Bag of Words, TF-IDF and text classification.
