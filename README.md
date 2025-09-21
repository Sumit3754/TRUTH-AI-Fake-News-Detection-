# Truth - Fake News Detection Webapp

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

## Gemini API Integration

This project also integrates with the Gemini API for advanced text analysis. The `gemini_integration.py` file handles the communication with the Gemini model to provide a deeper analysis of the text, including a confidence score and educational insights about potential misinformation techniques.

## Chrome Extension Setup

A Chrome extension is available in the `extension` directory to allow for quick analysis of text from any webpage.

### 1. Add Your API Key

Before you can use the extension, you must add your Gemini API key.

1.  Open the `extension/background.js` file.
2.  Find the line that says `const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY';`.
3.  Replace `'YOUR_GEMINI_API_KEY'` with your actual Gemini API key.

### 2. Install the Extension

1.  Open Google Chrome and navigate to `chrome://extensions`.
2.  Enable **"Developer mode"** using the toggle switch in the top-right corner.
3.  Click the **"Load unpacked"** button.
4.  Select the `extension` folder from this project's directory.

### 3. How to Use

Once installed, you can use the extension in two ways:

*   **Context Menu**: Highlight any text on a webpage, right-click, and select "Check for Misinformation". The extension's popup will open with the selected text ready to be analyzed.
*   **Popup**: Click on the extension's icon in your Chrome toolbar to open the popup. You can then paste any text directly into the text area and click "Analyze Text".

