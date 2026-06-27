import streamlit as st
import joblib

@st.cache_resource
def load_model():
    model = joblib.load('model/model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

st.title(" Email Spam Detector")
st.markdown("Write or paste your here - I m gonna tell you its Spam or Not!")

email_input = st.text_area("Write your Email Text here:", height=200)

if st.button(" Check !"):
    if email_input.strip() == "":
        st.warning("First Write Something here!")
    else:
        vec = vectorizer.transform([email_input])
        result = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0]

        if result == 1:
            st.error(f"It's  a SPAM !")
            st.error(f"Confidence: {round(max(confidence)*100, 2)}%")
        else:
            st.success(f" Not a SPAM  — Safe Email!")
            # st.success(f"Confidence: {round(max(confidence)*100, 2)}%")