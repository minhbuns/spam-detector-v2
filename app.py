import joblib
import streamlit as st
from preprocess import clean_text

@st.cache_resource
def load():
    return (joblib.load("model_phrase.joblib"), 
            joblib.load("model_word.joblib"),
            joblib.load("tdf_phrase.joblib"), 
            joblib.load("tdf_word.joblib"))

model_p, model_w, tdf_p, tdf_w = load()
             
st.title("Spam Detector V2")

st.text("Attention, all model runs on threshold 0.4 not 0.5 as V1")
model_choice = st.radio("We have 2 model, word and phrase, please choose 1.",
                 ['Word model', 'Phrase model'])

text = st.text_area("Paste an email:", height=200)

if st.button("Check") and text.strip() and model_choice:
    model = model_w if model_choice == 'Word model' else model_p
    tdf = tdf_w if model_choice == 'Word model' else tdf_p
    
    clean_input = tdf.transform(clean_text(text))
    p = model.predict_proba(clean_input)[0][1]
    if p >= 0.4:
        st.error(f"Spam ({p:.1%} probability)")
    else:
        st.success(f"Not spam ({1 - p:.1%} confidence)")
