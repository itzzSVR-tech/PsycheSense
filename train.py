import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib

# Load data (make sure mbti_1.csv is in the same folder)
df = pd.read_csv("mbti_1.csv")

# Join posts separated by ||| into one string
df['posts'] = df['posts'].apply(lambda x: " ".join(x.split("|||")))

X = df['posts']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=500))
])

# Calibrated classifier for probability outputs
calibrated_clf = CalibratedClassifierCV(pipeline, cv=3)
calibrated_clf.fit(X_train, y_train)

y_pred = calibrated_clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model
joblib.dump(calibrated_clf, "mbti_model_calibrated.pkl")
