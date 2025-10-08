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

5. (Optional) Run using Docker:
```bash
docker build -t mbti-streamlit-app .
docker run -p 8501:8501 mbti-streamlit-app
Open your browser and navigate to http://localhost:8501
```
