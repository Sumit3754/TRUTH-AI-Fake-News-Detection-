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
        stratify=data['fake']  # Ensures balanced split
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
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Root variables for consistent theming */
    :root {
        --primary-navy: #0B1426;
        --electric-blue: #00D4FF;
        --accent-blue: #1E40AF;
        --light-blue: #E0F2FE;
        --white: #FFFFFF;
        --gray-100: #F8FAFC;
        --gray-200: #E2E8F0;
        --gray-600: #475569;
        --gradient-primary: linear-gradient(135deg, #0B1426 0%, #1E40AF 50%, #00D4FF 100%);
        --gradient-secondary: linear-gradient(45deg, #00D4FF, #1E40AF);
    }
    
    /* Light mode variables */
    [data-theme="light"] {
        --primary-navy: #FFFFFF;
        --electric-blue: #1E40AF;
        --accent-blue: #00D4FF;
        --light-blue: #1E293B;
        --white: #0F172A;
        --gray-100: #0F172A;
        --gray-200: #1E293B;
        --gray-600: #64748B;
        --gradient-primary: linear-gradient(135deg, #FFFFFF 0%, #E0F2FE 50%, #BFDBFE 100%);
        --gradient-secondary: linear-gradient(45deg, #1E40AF, #00D4FF);
    }
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--primary-navy);
        color: var(--white);
        transition: all 0.3s ease;
    }
    
    /* Apply theme to body and html */
    html[data-theme="light"] .stApp,
    body[data-theme="light"] .stApp {
        background: var(--primary-navy);
        color: var(--white);
    }
    
    html[data-theme="dark"] .stApp,
    body[data-theme="dark"] .stApp {
        background: var(--primary-navy);
        color: var(--white);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--primary-navy);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--electric-blue);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Navigation Header */
    .nav-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: rgba(11, 20, 38, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .nav-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .logo {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--electric-blue);
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .logo-icon {
        width: 32px;
        height: 32px;
        background: var(--gradient-secondary);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    .theme-toggle {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 50%;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
    }
    
    .theme-toggle:hover {
        background: rgba(0, 212, 255, 0.2);
        transform: scale(1.1);
    }
    
    .theme-toggle span {
        font-size: 1.2rem;
        transition: transform 0.3s ease;
    }
    
    .theme-toggle:hover span {
        transform: rotate(20deg);
    }
    
    /* Hero Section */
    .hero-section {
        min-height: 120vh;
        background: var(--gradient-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 8rem 2rem 6rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Video Background */
    .video-background {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        overflow: hidden;
    }
    
    .video-background video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0.8;
        filter: brightness(1.1) contrast(1.1);
        transform: scale(1.05);
        animation: videoFloat 10s ease-in-out infinite;
    }
    
    @keyframes videoFloat {
        0%, 100% { 
            transform: scale(1.05) translateY(0px);
        }
        50% { 
            transform: scale(1.08) translateY(-10px);
        }
    }
    
    .video-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(11, 20, 38, 0.3) 0%, rgba(30, 64, 175, 0.2) 50%, rgba(0, 212, 255, 0.1) 100%);
        z-index: 1;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(30, 64, 175, 0.05) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite alternate;
        z-index: 2;
    }
    
    @keyframes pulse {
        0% { 
            transform: scale(1) rotate(0deg);
            opacity: 0.7;
        }
        100% { 
            transform: scale(1.05) rotate(2deg);
            opacity: 1;
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .hero-content {
        position: relative;
        z-index: 3;
        max-width: 800px;
    }
    
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--white) 0%, var(--electric-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #00D4FF, #FFFFFF, #00D4FF);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Poppins', sans-serif;
        min-height: 2rem;
        position: relative;
    }
    
    .hero-subtitle::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(0, 212, 255, 0.1), transparent);
        animation: shine 2s infinite;
        border-radius: 10px;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) skewX(-15deg); }
        100% { transform: translateX(200%) skewX(-15deg); }
    }
    
    .typing-animation {
        border-right: 2px solid var(--electric-blue);
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { border-color: var(--electric-blue); }
        51%, 100% { border-color: transparent; }
    }
    
    .cta-button {
        display: inline-block;
        padding: 1rem 2.5rem;
        background: var(--gradient-secondary);
        color: var(--white);
        text-decoration: none;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        margin: 1rem;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.4);
        color: var(--white);
        text-decoration: none;
    }
    
    /* Statistics Section */
    .stats-section {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 4rem;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        min-width: 150px;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--electric-blue);
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--light-blue);
        margin-top: 0.5rem;
    }
    
    /* Analysis Section */
    .analysis-section {
        background: transparent;
        border-radius: 0;
        padding: 2rem 1rem;
        margin: 2rem auto;
        max-width: 1000px;
        border: none;
    }
    
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: var(--electric-blue);
    }
    
    /* Form Styling */
    .stTextArea textarea {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 0 !important;
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 1rem 0 !important;
        font-size: 1rem !important;
        resize: vertical !important;
    }
    
    .stTextArea textarea:focus {
        border-bottom: 2px solid var(--electric-blue) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(224, 242, 254, 0.6) !important;
        font-style: italic !important;
    }
    
    .stTextArea > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stButton > button {
        background: var(--gradient-secondary) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
        }
        
        .stats-section {
            gap: 1rem;
        }
        
        .analysis-section {
            margin: 1rem 0.5rem;
            padding: 1rem 0.5rem;
        }
        
        .nav-content {
            padding: 0 1rem;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
        
        .video-background video {
            opacity: 0.5;
            filter: none;
            transform: scale(1.1);
        }
        
        .hero-section {
            min-height: 100vh;
            padding: 6rem 1rem 4rem;
        }
    }
    
    @media (max-width: 480px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            letter-spacing: 1px;
        }
        
        .analysis-section {
            margin: 0.5rem 0.25rem;
            padding: 0.5rem 0.25rem;
        }
        
        .section-title {
            font-size: 1.3rem;
        }
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #EF4444, #DC2626) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize theme in session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    
    # Navigation Header with Theme Toggle
    st.markdown("""
    <div class="nav-header">
        <div class="nav-content">
            <div class="logo">
                <div class="logo-icon">🛡️</div>
                TRUTH-AI
            </div>
            <div class="theme-toggle" onclick="toggleTheme()">
                <span id="theme-icon">🌙</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle in sidebar for mobile
    with st.sidebar:
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(f"{theme_icon} Switch to {'Light' if st.session_state.dark_mode else 'Dark'} Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 API Status")
        if st.button("Test Gemini Connection"):
            success, message = test_gemini_connection()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    # Apply theme based on session state
    theme_class = "" if st.session_state.dark_mode else 'data-theme="light"'
    
    # Load and encode video for Streamlit
    import base64
    import os
    
    def get_video_base64(video_path):
        try:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode()
                return f"data:video/mp4;base64,{video_base64}"
        except FileNotFoundError:
            return None
    
    # Get video data
    video_path = "truth-vid.mp4"
    video_data = get_video_base64(video_path)
    
    # Hero Section with theme support and video background
    if video_data:
        st.markdown(f"""
        <div {theme_class}>
            <div class="hero-section">
                <div class="video-background">
                    <video autoplay muted loop playsinline preload="auto">
                        <source src="{video_data}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div class="video-overlay"></div>
                <div class="hero-content">
                    <h1 class="hero-title">TRUTH-AI: Your Shield Against Misinformation</h1>
                    <div class="hero-subtitle">Detect. Learn. Protect.</div>
                    <div class="stats-section">
                        <div class="stat-item">
                            <span class="stat-number">24/7</span>
                            <div class="stat-label">Protection</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback without video if file not found
        st.markdown(f"""
        <div {theme_class}>
            <div class="hero-section">
                <div class="video-overlay"></div>
                <div class="hero-content">
                    <h1 class="hero-title">TRUTH-AI: Your Shield Against Misinformation</h1>
                    <div class="hero-subtitle">Detect. Learn. Protect.</div>
                    <div class="stats-section">
                        <div class="stat-item">
                            <span class="stat-number">24/7</span>
                            <div class="stat-label">Protection</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add JavaScript for theme functionality
    theme_js = f"""
    <script>
    // Set initial theme
    document.documentElement.setAttribute('data-theme', '{"dark" if st.session_state.dark_mode else "light"}');
    
    // Update theme icon
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {{
        themeIcon.textContent = '{"🌙" if st.session_state.dark_mode else "☀️"}';
    }}
    
    // Theme toggle function
    function toggleTheme() {{
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) {{
            themeIcon.textContent = newTheme === 'dark' ? '🌙' : '☀️';
        }}
    }}
    
    // Apply theme to body
    document.body.setAttribute('data-theme', '{"dark" if st.session_state.dark_mode else "light"}');
    </script>
    """
    
    st.markdown(theme_js, unsafe_allow_html=True)

    # Analysis Section
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🔍 Analyze Your News Article</h2>', unsafe_allow_html=True)
    
    # Add engaging slogan with effects
    st.markdown("""
    <div style="
        text-align: center;
        margin: 2rem 0;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(30, 64, 175, 0.1));
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
            animation: shimmer 3s infinite;
        "></div>
        <h3 style="
            font-family: 'Poppins', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--electric-blue);
            margin: 0;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
        ">
            💡 "In a world of information overload, let AI be your truth compass"
        </h3>
        <p style="
            font-size: 1rem;
            color: var(--light-blue);
            margin: 0.5rem 0 0 0;
            font-style: italic;
            opacity: 0.9;
        ">
            Paste any news article below and discover its authenticity in seconds
        </p>
    </div>
    
    <style>
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes glow {
        0% { text-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
        100% { text-shadow: 0 0 20px rgba(0, 212, 255, 0.6), 0 0 30px rgba(0, 212, 255, 0.4); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input for user to input news article
        user_input = st.text_area(
            "Enter your news article here:",
            placeholder="Paste the news article text you want to analyze for authenticity...",
            height=200
        )
    
    with col2:
        st.markdown("### ⚙️ Model Configuration")
        # Select vectorizer and classifier
        vectorizer, classifier = select_model()
        
        st.markdown("### 📊 Quick Stats")
        st.info("⚡ **Speed**: < 1 second")
        st.info("🔒 **Privacy**: Your data stays secure")
    
    # Initialize session state
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None

    # Center the check button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button("🔍 Analyze Article", use_container_width=True)

    # When user submits the input - FIXED
    if check_button and user_input.strip():
        st.session_state.user_input = user_input
        st.session_state.analysis_count += 1
        
        with st.spinner("🤖 Analyzing article authenticity..."):
            # Train the model and get the fitted vectorizer and accuracy
            fitted_vectorizer, clf, accuracy = train_model(data, vectorizer, classifier)
            
            # Store accuracy
            st.session_state.model_accuracy = accuracy
            
            # Vectorize the user input with the FITTED vectorizer
            input_vectorized = fitted_vectorizer.transform([st.session_state.user_input])
            
            # Predict the label of the input
            prediction = clf.predict(input_vectorized)
            
            # Store result in session state
            # prediction[0] will be 1 for FAKE, 0 for REAL
            st.session_state.result = int(prediction
