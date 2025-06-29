
# Emoji Predictor 😄➡️

A fun and simple machine learning project that predicts emojis from text using basic NLP and a Naive Bayes model.

## 🔍 Description
Given a short sentence, this model predicts the most appropriate emoji (e.g., 😊, 😢, 😠, 😂, 😨).

## 🧠 Model
- Model: Multinomial Naive Bayes
- Tools: Scikit-learn, CountVectorizer, joblib
- Pipeline: Text Vectorization + Classification

## 📊 Dataset
Created manually with 16 short emotional sentences labeled with emojis.

## 📂 Files Included
- `emoji_predictor_model.joblib` – Trained ML model
- `emoji_dataset.csv` – Training dataset
- `requirements.txt` – Required Python packages

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Load and use the model:

```python
import joblib
model = joblib.load("emoji_predictor_model.joblib")
print(model.predict(["I am really happy today"]))  # Output: ['😊']
```
