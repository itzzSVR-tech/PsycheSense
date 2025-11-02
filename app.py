import streamlit as st
import joblib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
from lime.lime_text import LimeTextExplainer  # type: ignore

# Load model
model = joblib.load("mbti_model_calibrated.pkl")
class_names = model.classes_

# MBTI Personality Type Descriptions
mbti_descriptions = {
    "INTJ": "The Architect - Strategic, independent, and insightful. You value logic, efficiency, and long-term planning.",
    "INTP": "The Thinker - Innovative, logical, and curious. You enjoy abstract thinking and solving complex problems.",
    "ENTJ": "The Commander - Bold, decisive, and strategic. You're a natural leader who excels at organizing and achieving goals.",
    "ENTP": "The Debater - Smart, curious, and quick-witted. You love intellectual challenges and exploring possibilities.",
    "INFJ": "The Advocate - Creative, insightful, and principled. You're passionate about helping others and making a difference.",
    "INFP": "The Mediator - Idealistic, loyal, and adaptable. You value authenticity and personal growth.",
    "ENFJ": "The Protagonist - Charismatic, inspiring, and natural-born leaders. You focus on helping others reach their potential.",
    "ENFP": "The Campaigner - Enthusiastic, creative, and sociable. You're a free spirit who loves exploring new possibilities.",
    "ISTJ": "The Logistician - Practical, fact-minded, and reliable. You value tradition, stability, and thoroughness.",
    "ISFJ": "The Protector - Warm, dedicated, and reliable. You're protective of those you care about and value security.",
    "ESTJ": "The Executive - Organized, practical, and decisive. You excel at managing people and implementing systems.",
    "ESFJ": "The Consul - Extraordinarily caring, social, and popular. You're always ready to protect loved ones.",
    "ISTP": "The Virtuoso - Bold, practical, and experimental. You're a master of all kinds of tools and hands-on activities.",
    "ISFP": "The Adventurer - Flexible, charming, and always ready to explore new possibilities. You value freedom and self-expression.",
    "ESTP": "The Entrepreneur - Smart, energetic, and perceptive. You're a risk-taker who thrives on action and immediate results.",
    "ESFP": "The Entertainer - Spontaneous, energetic, and people-focused. You love life, excitement, and making the most of every moment."
}

st.title("MBTI Personality Predictor")

st.write(
    """
    Enter your text and get a prediction of your MBTI personality type.
    You'll also see which words influenced the prediction.
    """
)

# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'lime_explanation' not in st.session_state:
    st.session_state.lime_explanation = None

user_input = st.text_area("Your text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
        st.session_state.prediction_results = None
        st.session_state.lime_explanation = None
    else:
        # Prediction and probabilities
        pred = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])[0]
        prob_dict = dict(zip(class_names, probs))
        
        # Store results in session state
        st.session_state.prediction_results = {
            'pred': pred,
            'prob_dict': prob_dict,
            'user_input': user_input
        }
        
        # LIME Explanation (compute and store)
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            user_input, model.predict_proba, num_features=20  # Increased features for more depth
        )
        
        pred_index = np.where(class_names == pred)[0][0]
        available_labels = exp.available_labels()
        
        if pred_index in available_labels:
            lime_features = exp.as_list(label=pred_index)
        else:
            lime_features = exp.as_list(label=available_labels[0])
        
        # Store expanded explanation data
        positive_features = [(f, w) for f, w in lime_features if w > 0]
        negative_features = [(f, w) for f, w in lime_features if w < 0]
        
        # Note: exp_html is stored for potential future use, but not currently displayed
        # Skip HTML generation to avoid compatibility issues with different LIME versions
        exp_html = None
        
        st.session_state.lime_explanation = {
            'features': lime_features,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'pred_index': pred_index,
            'exp_html': exp_html
        }

# Display results if they exist in session state
if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    pred = results['pred']
    prob_dict = results['prob_dict']
    
    st.success(f"**Predicted Personality:** {pred}")
    
    # Feature 1: Personality Type Insights
    st.write("### About Your Personality Type")
    if pred in mbti_descriptions:
        st.info(mbti_descriptions[pred])
    else:
        st.info(f"Your text suggests a {pred} personality type.")
    
    # Feature 2: Probability Visualization
    st.write("### Prediction Confidence")
    prob_df = pd.DataFrame({
        'Personality Type': list(prob_dict.keys()),
        'Probability': list(prob_dict.values())
    })
    prob_df = prob_df.sort_values('Probability', ascending=False)
    
    # Chart selection toggle
    chart_type = st.radio(
        "Select chart type to view:",
        ["Bar Chart", "Pie Chart", "Area Chart"],
        horizontal=True
    )
    
    # Display selected chart
    if chart_type == "Bar Chart":
        st.bar_chart(prob_df.set_index('Personality Type')['Probability'])
    elif chart_type == "Pie Chart":
        top_8_df = prob_df.head(8)
        pie_fig = px.pie(
            top_8_df,
            values='Probability',
            names='Personality Type',
            title='Probability Distribution (Top 8)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(pie_fig, use_container_width=True)
    elif chart_type == "Area Chart":
        st.area_chart(prob_df.set_index('Personality Type')['Probability'])
    
    # Display top 3 predictions
    top_3 = prob_df.head(3)
    st.write("**Top 3 Predictions:**")
    for idx, row in top_3.iterrows():
        percentage = row['Probability'] * 100
        st.write(f"- **{row['Personality Type']}**: {percentage:.1f}%")
    
    # Enhanced LIME Explanation
    if st.session_state.lime_explanation is not None:
        lime_data = st.session_state.lime_explanation
        
        st.write("### 🔍 Detailed Explanation: Key Words & Phrases")
        st.caption("Understanding which words and phrases influenced your personality prediction")
        
        # Summary statistics
        pos_count = len(lime_data['positive_features'])
        neg_count = len(lime_data['negative_features'])
        total_impact = sum(abs(w) for _, w in lime_data['features'])
        pos_impact = sum(w for _, w in lime_data['positive_features'])
        neg_impact = abs(sum(w for _, w in lime_data['negative_features']))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Supporting Features", pos_count)
        with col2:
            st.metric("Opposing Features", neg_count)
        with col3:
            st.metric("Positive Impact", f"{pos_impact:.3f}")
        with col4:
            st.metric("Negative Impact", f"{neg_impact:.3f}")
        
        # Visualization: Feature Importance Chart
        st.write("#### 📊 Feature Importance Visualization")
        
        # Prepare data for visualization
        all_features_df = pd.DataFrame(lime_data['features'], columns=['Feature', 'Weight'])
        all_features_df['Abs Weight'] = all_features_df['Weight'].abs()
        all_features_df = all_features_df.sort_values('Abs Weight', ascending=False).head(15)
        all_features_df['Color'] = all_features_df['Weight'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        
        # Horizontal bar chart showing feature importance
        fig = px.bar(
            all_features_df,
            x='Weight',
            y='Feature',
            orientation='h',
            color='Color',
            color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
            title='Top 15 Features Contributing to Prediction',
            labels={'Weight': 'Contribution Weight', 'Feature': 'Feature/Word'}
        )
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        
        # Positive Features (Supporting the prediction)
        st.write("#### ✅ Features Supporting Your Prediction")
        st.caption(f"These {pos_count} features contributed positively to the {pred} prediction:")
        
        pos_df = pd.DataFrame(lime_data['positive_features'], columns=['Feature', 'Weight'])
        pos_df = pos_df.sort_values('Weight', ascending=False)
        pos_df['Percentage'] = (pos_df['Weight'] / pos_impact * 100).round(2) if pos_impact > 0 else 0
        
        # Display as styled table
        if not pos_df.empty:
            max_weight = pos_df['Weight'].max()
            for idx, row in pos_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress_val = min(row['Weight'] / max_weight * 100, 100) if max_weight > 0 else 0
                    st.markdown(f"**{row['Feature']}**")
                    st.progress(progress_val / 100)
                with col2:
                    st.metric("", f"{row['Weight']:.4f}", delta=f"{row['Percentage']:.1f}%")
        
        # Negative Features (Opposing the prediction)
        st.write("#### ❌ Features Opposing Your Prediction")
        st.caption(f"These {neg_count} features contributed negatively to the {pred} prediction:")
        
        neg_df = pd.DataFrame(lime_data['negative_features'], columns=['Feature', 'Weight'])
        neg_df = neg_df.sort_values('Weight', ascending=True)  # Most negative first
        neg_df['Percentage'] = (abs(neg_df['Weight']) / neg_impact * 100).round(2) if neg_impact > 0 else 0
        
        if not neg_df.empty:
            min_weight = abs(neg_df['Weight'].min())
            for idx, row in neg_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    progress_val = min(abs(row['Weight']) / min_weight * 100, 100) if min_weight > 0 else 0
                    st.markdown(f"**{row['Feature']}**")
                    st.progress(progress_val / 100)
                with col2:
                    st.metric("", f"{row['Weight']:.4f}", delta=f"-{row['Percentage']:.1f}%")
        
        # Text Highlighting View
        st.write("#### 📝 Your Text with Highlighted Features")
        
        # Create a simple text highlighting visualization
        user_text = results['user_input']
        
        # Highlight positive features in green, negative in red
        highlighted_text = user_text
        pos_sorted = sorted(lime_data['positive_features'], key=lambda x: abs(x[1]), reverse=True)[:5]
        neg_sorted = sorted(lime_data['negative_features'], key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Simple highlighting using markdown
        st.markdown("**Text with top influential features:**")
        st.info(f"📝 *Original text: {len(user_text)} characters*")
        
        # Show which features appear in the text
        features_in_text = []
        all_top_features = pos_sorted[:3] + neg_sorted[:3]
        
        for feature, weight in all_top_features:
            if feature.lower() in user_text.lower():
                features_in_text.append((feature, weight))
        
        if features_in_text:
            st.markdown("**Top features found in your text:**")
            for feature, weight in features_in_text:
                color = "🟢" if weight > 0 else "🔴"
                st.markdown(f"{color} **{feature}** (impact: {weight:.4f})")
        
        # Detailed breakdown
        with st.expander("📋 View Complete Feature List (All Features)"):
            st.write("**All features ranked by absolute importance:**")
            detailed_df = pd.DataFrame(lime_data['features'], columns=['Feature', 'Weight'])
            detailed_df['Abs Weight'] = detailed_df['Weight'].abs()
            detailed_df = detailed_df.sort_values('Abs Weight', ascending=False)
            detailed_df['Impact'] = detailed_df['Weight'].apply(
                lambda x: '🟢 Positive' if x > 0 else '🔴 Negative'
            )
            st.dataframe(
                detailed_df[['Feature', 'Weight', 'Abs Weight', 'Impact']],
                use_container_width=True,
                hide_index=True
            )
        
        # Interpretation guide
        with st.expander("💡 How to Interpret These Results"):
            st.markdown("""
            **Understanding Feature Weights:**
            - **Positive weights (🟢)**: Words/phrases that support the predicted personality type
            - **Negative weights (🔴)**: Words/phrases that work against the predicted type
            
            **Impact Strength:**
            - Higher absolute values = stronger influence
            - Features are ranked by their contribution magnitude
            
            **What This Means:**
            - The model identified specific linguistic patterns in your text
            - Some words pushed the prediction toward your type, others away
            - The overall balance determined your final personality prediction
            """)
