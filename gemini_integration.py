"""
TRUTH-AI - Gemini Integration Module
AI-powered misinformation detection with educational insights
"""

import os
import google.generativeai as genai
import streamlit as st
from typing import Dict, List, Optional
import json
import time

class GeminiAnalyzer:
    """
    Google Gemini AI integration for advanced misinformation analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI client
        
        Args:
            api_key (str, optional): Gemini API key. If None, will try to get from environment
        """
        # Get API key from parameter, environment variable, or Streamlit secrets
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
        
        if not self.api_key:
            st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in environment variables.")
            return
            
        try:
            # Configure Gemini AI
            genai.configure(api_key=self.api_key)
            # Use the latest Gemini 2.0 Flash model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.is_configured = True
            print("✅ Successfully initialized Gemini 2.0 Flash model")
            
        except Exception as e:
            st.error(f"❌ Failed to configure Gemini AI: {str(e)}")
            self.is_configured = False
    
    def analyze_text(self, text: str, ml_prediction: str = None) -> Dict:
        """
        Analyze text for misinformation using Gemini AI
        
        Args:
            text (str): Text to analyze
            ml_prediction (str, optional): ML model prediction for context
            
        Returns:
            Dict: Analysis results with confidence, red flags, and educational insights
        """
        if not self.is_configured:
            return self._get_fallback_response()
            
        try:
            # Construct detailed prompt for misinformation analysis
            prompt = self._create_analysis_prompt(text, ml_prediction)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return self._parse_gemini_response(response.text)
            else:
                return self._get_fallback_response()
                
        except Exception as e:
            st.error(f"⚠️ Gemini API error: {str(e)}")
            return self._get_fallback_response()
    
    def _create_analysis_prompt(self, text: str, ml_prediction: str = None) -> str:
        """
        Create detailed prompt for Gemini AI analysis
        """
        context = f"ML Model Prediction: {ml_prediction}" if ml_prediction else ""
        
        prompt = f"""
        You are an expert misinformation detection analyst. Analyze the following text for potential misinformation and provide educational insights.

        {context}

        Text to analyze: "{text}"

        Please provide your analysis in this exact JSON format:
        {{
            "confidence_score": [number from 0-100],
            "risk_level": "[LOW/MEDIUM/HIGH]",
            "prediction": "[REAL/LIKELY_REAL/UNCERTAIN/LIKELY_FAKE/FAKE]",
            "red_flags": [
                {{
                    "flag": "specific red flag detected",
                    "explanation": "why this is concerning",
                    "severity": "[LOW/MEDIUM/HIGH]"
                }}
            ],
            "credibility_indicators": [
                {{
                    "indicator": "positive or negative indicator",
                    "type": "[POSITIVE/NEGATIVE]",
                    "explanation": "what this means"
                }}
            ],
            "educational_insights": [
                "Key learning point 1",
                "Key learning point 2",
                "Key learning point 3"
            ],
            "verification_suggestions": [
                "How to fact-check this type of content",
                "What sources to consult",
                "Red flags to watch for"
            ],
            "summary": "Brief explanation of why this content is likely real or fake"
        }}

        Focus on identifying:
        1. Vague or unnamed sources
        2. Emotional manipulation techniques
        3. Unverifiable claims
        4. Missing attribution or official confirmation
        5. Sensationalist language
        6. Logical inconsistencies
        7. Signs of financial or political manipulation

        Provide specific, actionable educational content that helps users become better at identifying misinformation.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse Gemini response and extract structured data
        """
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()
            
            # Find JSON content (sometimes Gemini adds extra text)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate and clean the parsed data
                return self._validate_response_data(parsed_data)
            else:
                # If no JSON found, create response from raw text
                return self._create_response_from_text(response_text)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, create response from raw text
            return self._create_response_from_text(response_text)
        except Exception as e:
            st.warning(f"⚠️ Error parsing Gemini response: {str(e)}")
            return self._get_fallback_response()
    
    def _validate_response_data(self, data: Dict) -> Dict:
        """
        Validate and ensure response data has required fields
        """
        validated = {
            "confidence_score": min(100, max(0, data.get("confidence_score", 75))),
            "risk_level": data.get("risk_level", "MEDIUM").upper(),
            "prediction": data.get("prediction", "UNCERTAIN").upper(),
            "red_flags": data.get("red_flags", []),
            "credibility_indicators": data.get("credibility_indicators", []),
            "educational_insights": data.get("educational_insights", []),
            "verification_suggestions": data.get("verification_suggestions", []),
            "summary": data.get("summary", "Analysis completed with moderate confidence."),
            "analysis_timestamp": time.time()
        }
        
        # Ensure risk level is valid
        if validated["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
            validated["risk_level"] = "MEDIUM"
            
        return validated
    
    def _create_response_from_text(self, text: str) -> Dict:
        """
        Create structured response from raw text when JSON parsing fails
        """
        # Simple keyword-based analysis for fallback
        confidence = 70
        risk_level = "MEDIUM"
        
        # Look for common misinformation indicators
        suspicious_phrases = [
            "unnamed sources", "officials refuse to comment", "shocking discovery",
            "doctors hate this", "they don't want you to know", "viral post",
            "forward this message", "share before it's deleted"
        ]
        
        red_flags = []
        for phrase in suspicious_phrases:
            if phrase.lower() in text.lower():
                red_flags.append({
                    "flag": f"Contains suspicious phrase: '{phrase}'",
                    "explanation": "This type of language is often used in misinformation",
                    "severity": "MEDIUM"
                })
                confidence = min(90, confidence + 10)
                risk_level = "HIGH"
        
        return {
            "confidence_score": confidence,
            "risk_level": risk_level,
            "prediction": "LIKELY_FAKE" if risk_level == "HIGH" else "UNCERTAIN",
            "red_flags": red_flags,
            "credibility_indicators": [],
            "educational_insights": [
                "Look for specific sources and official confirmation",
                "Be skeptical of sensational claims",
                "Cross-reference with multiple reliable sources"
            ],
            "verification_suggestions": [
                "Check official websites and press releases",
                "Look for reporting by established news organizations",
                "Verify any statistics or claims with original sources"
            ],
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "analysis_timestamp": time.time()
        }
    
    def _get_fallback_response(self) -> Dict:
        """
        Provide fallback response when Gemini is unavailable
        """
        return {
            "confidence_score": 50,
            "risk_level": "MEDIUM",
            "prediction": "UNCERTAIN",
            "red_flags": [],
            "credibility_indicators": [],
            "educational_insights": [
                "Always verify information from multiple sources",
                "Look for official confirmations and press releases",
                "Be cautious of emotionally charged language"
            ],
            "verification_suggestions": [
                "Check the original source of the information",
                "Look for corroboration from reliable news outlets",
                "Verify any claims with official authorities"
            ],
            "summary": "AI analysis temporarily unavailable. Please verify manually.",
            "analysis_timestamp": time.time(),
            "fallback": True
        }
    
    def get_risk_color(self, risk_level: str) -> str:
        """
        Get color code for risk level display
        """
        colors = {
            "LOW": "#28a745",      # Green
            "MEDIUM": "#ffc107",   # Yellow
            "HIGH": "#dc3545"      # Red
        }
        return colors.get(risk_level.upper(), "#6c757d")
    
    def format_confidence_display(self, confidence: int) -> str:
        """
        Format confidence score for display
        """
        if confidence >= 80:
            return f"🔴 High Confidence ({confidence}%)"
        elif confidence >= 60:
            return f"🟡 Medium Confidence ({confidence}%)"
        else:
            return f"🟢 Low Confidence ({confidence}%)"


# Utility functions for Streamlit integration
def display_gemini_results(analysis_results: Dict):
    """
    Display Gemini analysis results in Streamlit
    """
    if not analysis_results:
        st.error("❌ No analysis results available")
        return
    
    # Main results
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = analysis_results.get("confidence_score", 0)
        st.metric(
            label="AI Confidence Score",
            value=f"{confidence}%",
            delta=None
        )
    
    with col2:
        risk_level = analysis_results.get("risk_level", "MEDIUM")
        color = GeminiAnalyzer().get_risk_color(risk_level)
        st.markdown(
            f"<div style='padding: 10px; background-color: {color}; color: white; border-radius: 5px; text-align: center;'>"
            f"<strong>Risk Level: {risk_level}</strong></div>",
            unsafe_allow_html=True
        )
    
    # Summary
    summary = analysis_results.get("summary", "No summary available")
    st.subheader("🎯 Analysis Summary")
    st.write(summary)
    
    # Red Flags
    red_flags = analysis_results.get("red_flags", [])
    if red_flags:
        st.subheader("🚩 Red Flags Detected")
        for i, flag in enumerate(red_flags, 1):
            with st.expander(f"Red Flag {i}: {flag.get('flag', 'Unknown flag')}"):
                st.write(f"**Explanation:** {flag.get('explanation', 'No explanation available')}")
                st.write(f"**Severity:** {flag.get('severity', 'Unknown')}")
    
    # Educational Insights
    insights = analysis_results.get("educational_insights", [])
    if insights:
        st.subheader("📚 Educational Insights")
        for insight in insights:
            st.info(f"💡 {insight}")
    
    # Verification Suggestions
    suggestions = analysis_results.get("verification_suggestions", [])
    if suggestions:
        st.subheader("✅ How to Verify")
        for suggestion in suggestions:
            st.success(f"🔍 {suggestion}")


# Legacy function for backward compatibility
def analyze_text_with_gemini(text: str) -> Dict:
    """
    Legacy function for backward compatibility with existing code
    """
    analyzer = GeminiAnalyzer()
    if hasattr(analyzer, 'is_configured') and analyzer.is_configured:
        result = analyzer.analyze_text(text)
        # Convert new format to old format for compatibility
        return {
            'analysis': result.get('summary', 'Analysis completed'),
            'confidence_score': result.get('confidence_score', 50) / 100.0,  # Convert to 0-1 range
            'educational_insight': '\n'.join(result.get('educational_insights', ['General analysis completed']))
        }
    else:
        return {
            'analysis': 'Gemini API not configured',
            'confidence_score': 0.0,
            'educational_insight': 'Please configure Gemini API key'
        }

def list_available_models():
    """List all available Gemini models"""
    try:
        # Configure with API key
        api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
        if not api_key:
            return False, "No API key found"
        
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        return True, available_models
    except Exception as e:
        return False, f"Error listing models: {str(e)}"

def test_gemini_connection():
    """Test if Gemini API is working"""
    try:
        analyzer = GeminiAnalyzer()
        if hasattr(analyzer, 'is_configured') and analyzer.is_configured:
            try:
                response = analyzer.model.generate_content("Say 'Hello, Gemini is working!' and nothing else.")
                if response and response.text:
                    return True, f"✅ {response.text.strip()}"
                else:
                    return False, "Empty response from Gemini"
            except Exception as e:
                return False, f"API call failed: {str(e)}"
        else:
            return False, "Gemini model not properly initialized"
    except Exception as e:
        return False, f"Initialization failed: {str(e)}"


# Example usage and testing
def test_gemini_integration():
    """
    Test function for Gemini integration
    """
    analyzer = GeminiAnalyzer()
    
    test_text = """
    BREAKING: Scientists at IIT Delhi discover that drinking green tea with honey 
    for 7 days can boost immunity by 300%. Study shows zero COVID cases in test 
    group of 1000 people. Government officials refuse to comment.
    """
    
    results = analyzer.analyze_text(test_text, "FAKE")
    return results


if __name__ == "__main__":
    # Test the integration
    print("Testing Gemini Integration...")
    test_results = test_gemini_integration()
    print(json.dumps(test_results, indent=2))
