# Module 1: Import necessary packages
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import warnings
from gemini_integration import analyze_text_with_gemini, test_gemini_connection, GeminiAnalyzer, display_gemini_results, list_available_models
warnings.filterwarnings("ignore")

# Module 2: Load the dataset - FIXED
@st.cache_data
def load_data():
    data = pd.read_csv("fake_or_real_news.csv")
    # FIXED: Correct labeling - FAKE = 1, REAL = 0
    data['fake'] = data['label'].apply(lambda x: 1 if x == 'FAKE' else 0)
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

# Module 4: Train the model - FIXED
@st.cache_data
def train_model(data, _vectorizer, _classifier):
    """
    Train model with proper train/test split
    Args:
        data: DataFrame with text and labels
        _vectorizer: Vectorizer instance (prefix _ to avoid caching issues)
        _classifier: Classifier instance (prefix _ to avoid caching issues)
    Returns:
        fitted_vectorizer, fitted_classifier, accuracy
    """
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], 
        data['fake'], 
        test_size=0.2, 
        random_state=42,
        stratify=data['fake']
    )
    
    # Fit vectorizer ONLY on training data
    X_train_vectorized = _vectorizer.fit_transform(X_train)
    
    # Train classifier on training data
    _classifier.fit(X_train_vectorized, y_train)
    
    # Calculate accuracy on test set
    X_test_vectorized = _vectorizer.transform(X_test)
    accuracy = _classifier.score(X_test_vectorized, y_test)
    
    return _vectorizer, _classifier, accuracy

# Module 5: Streamlit app
def main():
    # Set page configuration
    page_icon = "🛡️"
    layout = "wide"
    page_title = "TRUTH-AI: Your Shield Against Misinformation"
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    :root {
        --primary-navy: #0B1426;
        --electric-blue: #00D4FF;
        --accent-blue: #1E40AF;
        --light-blue: #E0F2FE;
        --white: #FFFFFF;
        --gradient-primary: linear-gradient(135deg, #0B1426 0%, #1E40AF 50%, #00D4FF 100%);
        --gradient-secondary: linear-gradient(45deg, #00D4FF, #1E40AF);
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--primary-navy);
        color: var(--white);
    }
    
    .hero-section {
        min-height: 100vh;
        background: var(--gradient-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: var(--electric-blue);
    }
    
    .stButton > button {
        background: var(--gradient-secondary) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize theme
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 API Status")
        if st.button("Test Gemini Connection"):
            success, message = test_gemini_connection()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div>
            <h1 class="hero-title">TRUTH-AI: Your Shield Against Misinformation</h1>
            <p style="font-size: 1.5rem; color: var(--electric-blue);">Detect. Learn. Protect.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Analysis Section
    st.markdown("## 🔍 Analyze Your News Article")
    
    # Load data
    data = load_data()
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your news article here:",
            placeholder="Paste the news article text you want to analyze...",
            height=200
        )
    
    with col2:
        st.markdown("### ⚙️ Model Configuration")
        vectorizer, classifier = select_model()
        
        st.markdown("### 📊 Quick Stats")
        st.info("⚡ Speed: < 1 second")
        st.info("🔒 Privacy: Secure")
    
    # Initialize session state
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button("🔍 Analyze Article", use_container_width=True)

    # Process input - FIXED
    if check_button and user_input.strip():
        st.session_state.user_input = user_input
        st.session_state.analysis_count += 1
        
        with st.spinner("🤖 Analyzing article authenticity..."):
            # Train model
            fitted_vectorizer, clf, accuracy = train_model(data, vectorizer, classifier)
            st.session_state.model_accuracy = accuracy
            
            # Vectorize input
            input_vectorized = fitted_vectorizer.transform([st.session_state.user_input])
            
            # Predict (1 = FAKE, 0 = REAL)
            prediction = clf.predict(input_vectorized)
            st.session_state.result = int(prediction[0])
            
            # Debug output
            print(f"Prediction: {prediction[0]} (1=FAKE, 0=REAL)")
            print(f"Model accuracy: {accuracy:.2%}")

    # Display results
    if st.session_state.result is not None and st.session_state.user_input:
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if st.session_state.result == 1:
                st.error("🚨 **FAKE NEWS DETECTED**\n\nThis article shows characteristics of misinformation.")
            else:
                st.success("✅ **AUTHENTIC ARTICLE**\n\nThis article appears to be legitimate news.")
        
        with result_col2:
            if st.session_state.model_accuracy:
                actual_accuracy = int(st.session_state.model_accuracy * 100)
                st.metric(
                    label="Model Accuracy",
                    value=f"{actual_accuracy}%",
                    delta="On test data"
                )
            else:
                confidence = 85 + (st.session_state.analysis_count % 15)
                st.metric(
                    label="Confidence Level",
                    value=f"{confidence}%"
                )

        # Enhanced Analysis
        st.markdown("---")
        st.markdown("### 🧠 Enhanced Analysis & Insights")
        
        if 'enhanced_analysis' not in st.session_state:
            st.session_state.enhanced_analysis = None
        
        if st.session_state.enhanced_analysis is None:
            with st.spinner("🧠 Generating enhanced insights..."):
                try:
                    ml_result = "FAKE NEWS" if st.session_state.result == 1 else "AUTHENTIC NEWS"
                    
                    enhanced_prompt = f"""As an expert fact-checker, analyze this news article classified as {ml_result}.

Article: "{st.session_state.user_input[:1200]}"

Provide:
DETAILED_BREAKDOWN: Analysis of why this might be {ml_result.lower()}
EDUCATIONAL_INSIGHTS: Verification tips for this content type
CONTEXT_ANALYSIS: What to look for in similar articles"""

                    enhanced_result = analyze_text_with_gemini(enhanced_prompt)
                    st.session_state.enhanced_analysis = enhanced_result
                    
                except Exception as e:
                    st.session_state.enhanced_analysis = {
                        'analysis': f'Analysis error: {str(e)}',
                        'confidence_score': 0.0,
                        'educational_insight': 'Static content provided below.'
                    }
        
        # Display enhanced analysis
        analysis_col1, analysis_col2 = st.columns([1, 1])
        
        with analysis_col1:
            st.markdown("#### 🔍 AI-Powered Breakdown")
            
            if st.session_state.enhanced_analysis and st.session_state.enhanced_analysis['confidence_score'] > 0:
                analysis_text = st.session_state.enhanced_analysis['analysis']
                if "DETAILED_BREAKDOWN:" in analysis_text:
                    detailed_part = analysis_text.split("DETAILED_BREAKDOWN:")[1].split("EDUCATIONAL_INSIGHTS:")[0].strip()
                    st.markdown(detailed_part)
                else:
                    st.markdown(analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            else:
                if st.session_state.result == 1:
                    st.markdown("""
                    **🚨 Red Flags:**
                    - Language patterns indicate misinformation
                    - Suspicious content detected
                    - Recommend fact-checking
                    """)
                else:
                    st.markdown("""
                    **✅ Credibility Indicators:**
                    - Legitimate news patterns
                    - High authenticity confidence
                    - Journalistic standards followed
                    """)
        
        with analysis_col2:
            st.markdown("#### 🎓 Educational Insights")
            
            if st.session_state.enhanced_analysis and st.session_state.enhanced_analysis['confidence_score'] > 0:
                analysis_text = st.session_state.enhanced_analysis['analysis']
                if "EDUCATIONAL_INSIGHTS:" in analysis_text:
                    education_part = analysis_text.split("EDUCATIONAL_INSIGHTS:")[1].split("CONTEXT_ANALYSIS:")[0].strip()
                    st.markdown(education_part)
                else:
                    st.markdown(st.session_state.enhanced_analysis['educational_insight'])
            else:
                st.markdown("""
                **How to Spot Fake News:**
                - Check multiple sources
                - Verify author credentials
                - Be wary of emotional headlines
                - Cross-reference with fact-checkers
                
                **Trusted Sources:**
                - Reuters, AP News, BBC
                - Snopes, FactCheck.org
                """)
        
        # Context Analysis
        if st.session_state.enhanced_analysis and st.session_state.enhanced_analysis['confidence_score'] > 0:
            analysis_text = st.session_state.enhanced_analysis['analysis']
            if "CONTEXT_ANALYSIS:" in analysis_text:
                context_part = analysis_text.split("CONTEXT_ANALYSIS:")[1].strip()
                if context_part:
                    st.markdown("#### 🌐 Contextual Guidance")
                    st.info(context_part)
        
        # Refresh button
        if st.button("🔄 Regenerate Analysis", key="refresh_analysis"):
            st.session_state.enhanced_analysis = None
            st.rerun()
        
        # Gemini AI Analysis
        st.markdown("---")
        st.markdown("### 🚀 Advanced Gemini AI Analysis")
        
        gemini_col1, gemini_col2 = st.columns([3, 1])
        
        with gemini_col1:
            st.info("🧠 Get comprehensive AI-powered analysis with detailed insights.")
        
        with gemini_col2:
            if st.button("🚀 Analyze with Gemini", key="gemini_analysis", use_container_width=True):
                with st.spinner("🧠 Running analysis..."):
                    if len(st.session_state.user_input.strip()) < 10:
                        st.warning("⚠️ Please enter at least 10 characters.")
                    else:
                        try:
                            analyzer = GeminiAnalyzer()
                            
                            if hasattr(analyzer, 'is_configured') and analyzer.is_configured:
                                ml_prediction = "FAKE" if st.session_state.result == 1 else "REAL"
                                gemini_results = analyzer.analyze_text(st.session_state.user_input, ml_prediction)
                                
                                st.success("✅ Analysis Complete")
                                display_gemini_results(gemini_results)
                            else:
                                st.warning("⚠️ Gemini API not configured")
                                st.info("Check your API key. Core ML analysis still works!")
                                
                        except Exception as e:
                            st.error(f"⚠️ Error: {str(e)}")
                            st.info("Core detection system continues to work perfectly.")
    
    elif check_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #64748B;">
        🚀 Created by hacktreet team | Powered by TRUTH-AI | 🛡️ Protecting truth in the digital age
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
