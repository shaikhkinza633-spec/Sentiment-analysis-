import streamlit as st
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")


with open("model", "rb") as file:
    model = pickle.load(file)

with open("bow", "rb") as file:
    bow = pickle.load(file)


no_stopword = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

review = st.text_area("Enter your movie review")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        
        lowercase = review.lower()


        html_remove = re.sub(r"<.*?>", "", lowercase)

        
        url_remove = re.sub(
            r"https?://\S+|www\.\S+", "", html_remove
        )

    
        punc_remove = re.sub(
            r"[^a-zA-Z\s]", " ", url_remove
        )

        
        word_tokenize = punc_remove.split()

        
        no_stopword_review = [
            word for word in word_tokenize
            if word not in no_stopword
        ]

        
        lemmatization = " ".join(
            lemmatizer.lemmatize(word)
            for word in no_stopword_review
        )

        
        vector = bow.transform([lemmatization])

        
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")
