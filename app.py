import joblib
import streamlit as st
from preprocess import clean_text
def load():
    return (
        joblib.load("model_email.joblib"),
        joblib.load("model_sms.joblib"),
        joblib.load("vec_email.joblib"),
        joblib.load("vec_sms.joblib")
    )

model_e, model_s, vec_e, vec_s = load()

st.title("Spam Detector V2.5")

model_choice = st.radio("We have 2 model, email and sms, please choose 1.",
                 ['Email model', 'SMS model'])

text = st.text_area("Paste an email:", height=200)

if st.button("Check") and text.strip() and model_choice:
    model = model_e if model_choice == 'Email model' else model_s
    vec = vec_e if model_choice == 'Email model' else vec_s
    clean_input = vec.transform([clean_text(text)]) 
    p = model.predict_proba(clean_input)[0][1] 
    if p >= 0.5:
        st.error(f"Spam ({p:.1%} probability)")
    else:
        st.success(f"Not spam ({1 - p:.1%} confidence)")
