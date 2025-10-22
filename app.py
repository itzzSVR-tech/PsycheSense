import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load model
model = joblib.load("mbti_model_calibrated.pkl")
class_names = model.classes_

st.title("Advanced MBTI Personality Predictor")

st.write(
    """
Enter your text and get a prediction of your MBTI personality type.
You'll also see which words influenced the prediction.
"""
)

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
        exp = explainer.explain_instance(
            user_input, model.predict_proba, num_features=10
        )

        # Get the index of the predicted class
        pred_index = np.where(class_names == pred)[0][0]

        st.write("### Explanation of key words:")
        # Use exp.as_list() with the correct label, or use available_labels() to check
        available_labels = exp.available_labels()
        if pred_index in available_labels:
            for feature, weight in exp.as_list(label=pred_index):
                st.write(f"{feature}: {weight:.3f}")
        else:
            # Fall back to the first available label (top prediction)
            for feature, weight in exp.as_list(label=available_labels[0]):
                st.write(f"{feature}: {weight:.3f}")
