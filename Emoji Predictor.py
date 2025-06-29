import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Dataset
data = {
    "text": [
        "I am so happy today", "I feel very sad", "I am angry right now",
        "This is amazing", "I hate this", "I love you", "I am scared",
        "This is so funny", "I am crying", "This is terrible", "I’m excited",
        "That was disappointing", "You make me smile", "I'm frustrated",
        "That joke was hilarious", "I'm feeling nervous"
    ],
    "emoji": [
        "😊", "😢", "😠", "😊", "😠", "😊", "😨", "😂", "😢", "😠", "😊", "😢", "😊", "😠", "😂", "😨"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emoji"], test_size=0.2, random_state=42)

# Create model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "emoji_predictor_model.joblib")

# Save dataset
df.to_csv("emoji_dataset.csv", index=False)

print("\nModel and dataset saved successfully.")