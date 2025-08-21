Skip to content
You said:
Provide a Personality Prediction Project including a frontend and backend using Python(Streamlit) and ML Algorithms using the MBTI Dataset.
ChatGPT said:
Sure! I’ll guide you through building a simple Personality Prediction Project using the MBTI dataset. We'll use:

Backend: Python with scikit-learn for training the ML model.

Frontend: Streamlit for a user-friendly web interface.

Project Overview:
Dataset: MBTI (Myers-Briggs Type Indicator) Dataset — includes text posts and corresponding personality types.

Goal: Predict the MBTI personality type based on user input text.

ML model: We'll use a text classification pipeline (TF-IDF + Logistic Regression).

Frontend: Streamlit app where users enter text and get personality prediction.

Step 1: Get the MBTI Dataset
You can download the dataset from Kaggle or other sources; a popular one is here.

For simplicity, the dataset looks like this (CSV):

type	posts
INFJ	"I'm always introspective and... etc."
Step 2: Backend — Train the Model
2.1 Data Preparation & Training Script
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("mbti_1.csv")

# Combine all posts into one string
df['posts'] = df['posts'].apply(lambda x: " ".join(x.split("|||")))

# Split data
X = df['posts']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=500))
])

# Train
pipeline.fit(X_train, y_train)

# Test accuracy
y_pred = pipeline.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model
joblib.dump(pipeline, "mbti_model.pkl")
Step 3: Frontend — Streamlit App
import streamlit as st
import joblib

# Load the trained model
model = joblib.load("mbti_model.pkl")

st.title("MBTI Personality Type Predictor")

st.write("""
Enter your text (e.g., social media posts, thoughts) and the model will predict your MBTI personality type.
""")

user_input = st.text_area("Enter text here:", height=200)

if st.button("Predict Personality Type"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        st.success(f"Predicted MBTI Personality Type: **{prediction}**")
    else:
        st.warning("Please enter some text to predict.")
Step 4: How to Run
Save the training script as train.py and run it:

python train.py
This creates mbti_model.pkl.

Save the Streamlit app as app.py.

Run Streamlit:

streamlit run app.py
Optional Enhancements
Use a more powerful classifier (Random Forest, XGBoost, or even transformer-based embeddings).

Add confidence scores or probabilities.

Show explanations (e.g., using SHAP or LIME).

Allow input of multiple posts for better accuracy.

If you want, I can also help with a more advanced version or containerizing this app. Would you like me to help with that?

You said:
Yes
ChatGPT said:
Awesome! Let’s step it up with a more advanced version of your MBTI Personality Prediction app. Here’s what I suggest adding:

Advanced Enhancements:
Use TF-IDF + Logistic Regression but also show prediction probabilities.

Add explanation with LIME (Local Interpretable Model-agnostic Explanations) to show which words influenced the prediction.

Package the app in a Docker container so it’s portable and easy to deploy anywhere.

Step-by-step guide
Step 1: Upgrade Training Script (with probability calibration)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("mbti_1.csv")
df['posts'] = df['posts'].apply(lambda x: " ".join(x.split("|||")))

X = df['posts']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=500))
])

# Wrap LogisticRegression with calibration for probabilities
calibrated_clf = CalibratedClassifierCV(pipeline, cv=3)
calibrated_clf.fit(X_train, y_train)

y_pred = calibrated_clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

joblib.dump(calibrated_clf, "mbti_model_calibrated.pkl")
Step 2: Streamlit App with Probability + LIME
Install required packages:

pip install streamlit joblib lime scikit-learn
Create app.py:

import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load model
model = joblib.load("mbti_model_calibrated.pkl")

# Class names
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
Step 3: Create Dockerfile
Create a file named Dockerfile in your project directory:

# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY app.py .
COPY mbti_model_calibrated.pkl .

# Expose port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
Step 4: Create requirements.txt
streamlit
scikit-learn
joblib
lime
numpy
pandas
Step 5: Build and Run Docker Container
In your terminal:

docker build -t mbti-streamlit-app .
docker run -p 8501:8501 mbti-streamlit-app
Then open your browser at http://localhost:8501 to see the app.

Optional: I can help you automate training, save model, build Docker, or even deploy it on cloud (Heroku, AWS).
Would you like me to prepare the complete zipped project structure or help with deploying it?

You said:
Yes
ChatGPT said:
Great! Let me prepare a complete project structure for you, with all the necessary files so you can just clone/run it locally or build the Docker container easily.
