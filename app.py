import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

port_stem = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter news text to check whether it is **Fake or Real**")

news_input = st.text_area("Enter News Content", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter news text")
    else:
        processed = preprocess(news_input)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.success("✅ This is REAL News")
        else:
            st.error("❌ This is FAKE News")
