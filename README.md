# PsycheSense: MBTI Personality Prediction Project

## Setup

1. Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

2. Extract the MBTI dataset `mbti_1.csv` from `mbti_1.zip` and place it in the project root folder.

3. Train the model:
```bash
python train.py
```

4. Run the Streamlit app locally:
```bash
streamlit run app.py
```

5. Sample Paragraph for Checking

`I often find myself lost in thought, wandering through the labyrinth of my own imagination. The world outside can be so loud and demanding, but in the quiet corners of my mind, I can explore endless possibilities and connect with my deepest feelings. I'm not one for big crowds or superficial conversations; I'd much rather have a meaningful, one-on-one discussion about dreams, ideas, and the mysteries of the universe. I feel a strong sense of empathy for others and often find myself absorbing their emotions, which can be both a blessing and a curse. My decisions are guided by my values and what feels right in my heart, even if it's not the most logical or practical choice. I find beauty in the unconventional and am always searching for the magic in everyday life.`

6. (Optional) Run using Docker:
```bash
docker build -t mbti-streamlit-app .
docker run -p 8501:8501 mbti-streamlit-app
Open your browser and navigate to http://localhost:8501
```
