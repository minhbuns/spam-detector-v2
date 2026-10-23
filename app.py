import joblib
import streamlit as st
from preprocess import clean_text
for name in [
    "vec_email.joblib",
    "vec_sms.joblib",
    "model_email.joblib",
    "model_sms.joblib",
]:
    obj = joblib.load(name)
    print(f"{name}: {type(obj)}")

st.title("Spam Detector V2.5")

model_choice = st.radio("We have 2 model, email and sms, please choose 1.",
                 ['Email model', 'SMS model'])

text = st.text_area("Paste an email:", height=200)

if st.button("Check") and text.strip() and model_choice:
    model = model_email if model_choice == 'Email model' else model_sms
    vec = vec_email if model_choice == 'Email model' else vec_sms
    clean_input = vec.transform([clean_text(text)])
    p = model.predict_proba(clean_input)[0][1]
    if p >= 0.5:
        st.error(f"Spam ({p:.1%} probability)")
    else:
        st.success(f"Not spam ({1 - p:.1%} confidence)")
