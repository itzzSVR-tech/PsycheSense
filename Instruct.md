# Complete Project Structure
```
mbti-personality-predictor/
│
├── mbti_1.csv                # The MBTI dataset (you'll need to download this)
├── train.py                  # Training script
├── app.py                    # Streamlit app
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container setup
└── README.md                 # Instructions & info
```

## File contents

1. `train.py`

```python
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
```

2. `app.py`

```python
import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load model
model = joblib.load("mbti_model_calibrated.pkl")
class_names = model.classes_

st.title("Advanced MBTI Personality Predictor")

st.write("""
Enter your text and get a prediction of your MBTI personality type.
You'll also see which words influenced the prediction.
""")

user_input = st.text_area("Your text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        # Prediction and probabilities
        pred = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])[0]
        prob_dict = dict(zip(class_names, probs))
        
        st.success(f"**Predicted Personality:** {pred}")
        st.write("Prediction probabilities:")
        st.write(prob_dict)
        
        # LIME Explanation
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(user_input, model.predict_proba, num_features=10)
        
        st.write("### Explanation of key words:")
        for feature, weight in exp.as_list(label=np.where(class_names == pred)[0][0]):
            st.write(f"{feature}: {weight:.3f}")
```

3. `requirements.txt`

```txt
streamlit
scikit-learn
joblib
lime
numpy
pandas
```

4. `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY mbti_model_calibrated.pkl .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

5. README.md
```markdown
# MBTI Personality Prediction Project

## Setup

1. Download the MBTI dataset `mbti_1.csv` and place it in the project root folder.

2. Train the model:

```bash
python train.py
```
3. Run the Streamlit app locally:

```bash
streamlit run app.py
```
4. (Optional) Run using Docker:

```bash
docker build -t mbti-streamlit-app .
docker run -p 8501:8501 mbti-streamlit-app
Open your browser and navigate to http://localhost:8501
```

## Notes

The model uses TF-IDF + Logistic Regression.

Prediction includes probabilities and LIME explanations for interpretability.

---

# What you need to do now

1. Download the MBTI dataset CSV (I recommend Kaggle: https://www.kaggle.com/datasnaek/mbti-type)
2. Place `mbti_1.csv` in the project folder.
3. Run `python train.py` to train and save the model.
4. Run `streamlit run app.py` to launch the app.
5. Or build and run Docker container as described.

---

