# Fake News Detection Webapp

## Description

This Streamlit app is designed to detect whether a news article is likely fake or real based on its content. It allows users to input a news article, select a vectorizer and classifier, and then predicts the authenticity of the article.

## Modules

### Module 1: Import necessary packages

- `streamlit`: For creating the web application.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `sklearn`: For machine learning functionalities.
- `warnings`: For ignoring warnings.
- `streamlit_lottie`: For displaying Lottie animations.

### Module 2: Load the dataset

- Loads the dataset containing fake and real news articles from a CSV file.
- Converts the labels to binary format (0 for real, 1 for fake).

### Module 3: Select Vectorizer and Classifier

- Allows users to select a vectorizer (TF-IDF or Bag of Words) and a classifier (Linear SVM or Naive Bayes) via the sidebar.

### Module 4: Train the model

- Trains the selected classifier model using the chosen vectorizer and the loaded dataset.
- Caches the trained model for faster access.

### Module 5: Streamlit app

- Sets page configuration including title, icon, and layout.
- Displays the title and a Lottie animation.
- Hides the Streamlit style for a cleaner interface.
- Provides a text area for users to input news articles.
- Upon clicking the "Check" button, predicts the authenticity of the input news article using the trained model and displays the result.

## Usage

- Run the Streamlit app using the command: `streamlit run main.py --client.showErrorDetails=false` to remove cache error messages on the Streamlit interface.
- Input a news article into the text area.
- Select a vectorizer and classifier from the sidebar.
- Click the "Check" button to see the prediction result.

<img width="959" alt="image" src="https://github.com/SmridhVarma/Fake-News-Detection/assets/103480022/532e7401-1562-4369-aa90-35bfed044767">

