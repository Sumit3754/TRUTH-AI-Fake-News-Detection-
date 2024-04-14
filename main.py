# Module 1: Import necessary packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import warnings
import streamlit_lottie
warnings.filterwarnings("ignore")

# Module 2: Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("fake_or_real_news.csv")
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    return data

# Module 3: Select Vectorizer and Classifier
def select_model():
    vectorizer_type = st.sidebar.selectbox("Select Vectorizer", ["TF-IDF", "Bag of Words"])
    classifier_type = st.sidebar.selectbox("Select Classifier", ["Linear SVM", "Naive Bayes"])
    
    vectorizer = None
    if vectorizer_type == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    elif vectorizer_type == "Bag of Words":
        vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
    
    classifier = None
    if classifier_type == "Linear SVM":
        classifier = LinearSVC()
    elif classifier_type == "Naive Bayes":
        classifier = MultinomialNB()
    
    return vectorizer, classifier

# Module 4: Train the model
@st.cache
def train_model(data, vectorizer, classifier):
    x_vectorized = vectorizer.fit_transform(data['text'])
    clf = classifier.fit(x_vectorized, data['fake'])
    return clf

# Module 5: Streamlit app
def main():
    # Set page configuration
    page_icon = ":metro:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    layout = "wide"
    page_title = "Fake News Detection"
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    
    # Streamlit app
    st.title(page_title + " " + page_icon)
    st.lottie("https://lottie.host/bd0c4818-c5a6-4e42-b407-746bc448c2c7/ipVUdgFncO.json", width=200, height=200)

    # --- HIDE STREAMLIT STYLE ---
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Load data
    data = load_data()
    
    # Select vectorizer and classifier
    vectorizer, classifier = select_model()
    
    # Text input for user to input news article
    user_input = st.text_area("Enter your news article here:")
    
    # When user submits the input
    if st.button("Check"):
        # Train the model
        clf = train_model(data, vectorizer, classifier)
        
        # Vectorize the user input
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict the label of the input
        prediction = clf.predict(input_vectorized)
        
        # Convert prediction to integer for interpretation
        result = int(prediction[0])
        
        # Display the result
        if result == 1:
            st.error("This news article is likely fake!")
        else:
            st.success("This news article seems to be real.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
st.markdown("**Created with enthusiasm by SuperSam**")
##run with command streamlit run main.py --client.showErrorDetails=false to remove cache error message on streamlit interface
