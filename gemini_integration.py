"""
TRUTH-AI - Gemini Integration Module
AI-powered misinformation detection with educational insights + verification links
"""

import os
import google.generativeai as genai
import streamlit as st
from typing import Dict, Optional
import json
import time


class GeminiAnalyzer:
    """
    Google Gemini AI integration for advanced misinformation analysis
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI client
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')

        if not self.api_key:
            st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in environment variables.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.is_configured = True
            print("✅ Successfully initialized Gemini 2.0 Flash model")
        except Exception as e:
            st.error(f"❌ Failed to configure Gemini AI: {str(e)}")
            self.is_configured = False

    def analyze_text(self, text: str, ml_prediction: str = None) -> Dict:
        """
        Analyze text for misinformation using Gemini AI
        """
        if not self.is_configured:
            return self._get_fallback_response()

        try:
            prompt = self._create_analysis_prompt(text, ml_prediction)
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
        Create detailed prompt for Gemini AI analysis (extended to request verification links)
        """
        context = f"ML Model Prediction: {ml_prediction}" if ml_prediction else ""

        prompt = f"""
        You are an expert misinformation detection analyst. Analyze the following text for potential misinformation and provide educational insights.

        {context}

        Text to analyze: "{text}"

        Please provide your analysis in this exact JSON format (valid JSON only, avoid commentary outside JSON):
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
                "Key learning point 2"
            ],
            "verification_suggestions": [
                "How to fact-check this type of content",
                "What sources to consult"
            ],
            "verification_links": [
              {{
                "title": "Short descriptive title",
                "url": "https://example.com/source-article",
                "type": "[OFFICIAL_SITE/NEWS/RESEARCH/FACTCHECK]",
                "note": "one-line reason why this link is relevant"
              }}
            ],
            "verification_notes": "Short instructions on how to verify, or summary of key linked sources",
            "summary": "Brief explanation of why this content is likely real or fake"
        }}

        IMPORTANT:
        - Prefer authoritative sources for links (official agency sites, major news orgs, academic journals, recognized fact-checkers).
        - Provide up to 5 verification links, each with title, url, type and a one-line note.
        - If you cannot find direct sources, return an empty array for verification_links and explain which credible sources to check in verification_notes.
        - Keep all output strictly valid JSON; do not include text before/after the JSON object.
        """
        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse Gemini response and extract structured data
        """
        try:
            response_text = response_text.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                return self._validate_response_data(parsed_data)
            else:
                return self._create_response_from_text(response_text)

        except json.JSONDecodeError:
            return self._create_response_from_text(response_text)
        except Exception as e:
            st.warning(f"⚠️ Error parsing Gemini response: {str(e)}")
            return self._get_fallback_response()

    def _validate_response_data(self, data: Dict) -> Dict:
        """
        Validate and ensure response data has required fields (now supports verification_links)
        """
        validated = {
            "confidence_score": min(100, max(0, data.get("confidence_score", 75))),
            "risk_level": data.get("risk_level", "MEDIUM").upper(),
            "prediction": data.get("prediction", "UNCERTAIN").upper(),
            "red_flags": data.get("red_flags", []),
            "credibility_indicators": data.get("credibility_indicators", []),
            "educational_insights": data.get("educational_insights", []),
            "verification_suggestions": data.get("verification_suggestions", []),
            "verification_links": data.get("verification_links", []),
            "verification_notes": data.get("verification_notes", ""),
            "summary": data.get("summary", "Analysis completed with moderate confidence."),
            "analysis_timestamp": time.time()
        }

        if validated["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
            validated["risk_level"] = "MEDIUM"

        if isinstance(validated["verification_links"], list):
            sanitized_links = []
            for link in validated["verification_links"][:5]:
                if isinstance(link, dict) and "url" in link:
                    sanitized_links.append({
                        "title": link.get("title", link.get("url")),
                        "url": link.get("url"),
                        "type": link.get("type", "NEWS"),
                        "note": link.get("note", "")
                    })
            validated["verification_links"] = sanitized_links

        return validated

    def _create_response_from_text(self, text: str) -> Dict:
        """
        Create structured response from raw text when JSON parsing fails
        """
        confidence = 70
        risk_level = "MEDIUM"
        suspicious_phrases = [
            "unnamed sources", "officials refuse to comment", "shocking discovery",
            "they don't want you to know", "viral post", "share before it's deleted"
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
                "Look for reporting by established news organizations"
            ],
            "verification_links": [],
            "verification_notes": "",
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
                "Look for official confirmations and press releases"
            ],
            "verification_suggestions": [
                "Check the original source of the information",
                "Look for corroboration from reliable news outlets"
            ],
            "verification_links": [],
            "verification_notes": "",
            "summary": "AI analysis temporarily unavailable. Please verify manually.",
            "analysis_timestamp": time.time(),
            "fallback": True
        }

    def get_risk_color(self, risk_level: str) -> str:
        colors = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}
        return colors.get(risk_level.upper(), "#6c757d")

    def format_confidence_display(self, confidence: int) -> str:
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

    col1, col2 = st.columns(2)

    with col1:
        confidence = analysis_results.get("confidence_score", 0)
        st.metric(label="AI Confidence Score", value=f"{confidence}%")

    with col2:
        risk_level = analysis_results.get("risk_level", "MEDIUM")
        color = GeminiAnalyzer().get_risk_color(risk_level)
        st.markdown(
            f"<div style='padding: 10px; background-color: {color}; color: white; border-radius: 5px; text-align: center;'>"
            f"<strong>Risk Level: {risk_level}</strong></div>",
            unsafe_allow_html=True
        )

    st.subheader("🎯 Analysis Summary")
    st.write(analysis_results.get("summary", "No summary available"))

    red_flags = analysis_results.get("red_flags", [])
    if red_flags:
        st.subheader("🚩 Red Flags Detected")
        for i, flag in enumerate(red_flags, 1):
            with st.expander(f"Red Flag {i}: {flag.get('flag', 'Unknown flag')}"):
                st.write(f"**Explanation:** {flag.get('explanation', '')}")
                st.write(f"**Severity:** {flag.get('severity', '')}")

    insights = analysis_results.get("educational_insights", [])
    if insights:
        st.subheader("📚 Educational Insights")
        for insight in insights:
            st.info(f"💡 {insight}")

    suggestions = analysis_results.get("verification_suggestions", [])
    if suggestions:
        st.subheader("✅ How to Verify")
        for suggestion in suggestions:
            st.success(f"🔍 {suggestion}")

    links = analysis_results.get("verification_links", [])
    if links:
        st.subheader("🔗 Verification Resources")
        for link in links:
            st.write(f"- [{link.get('title')}]({link.get('url')}) — {link.get('type')} — {link.get('note')}")

    notes = analysis_results.get("verification_notes", "")
    if notes:
        st.info(f"📝 Verification Notes: {notes}")


# Test integration
if __name__ == "__main__":
    analyzer = GeminiAnalyzer()
    test_text = """
    BREAKING: Scientists at IIT Bombay develop new water purification tech,
    published in Nature Materials. ISRO also launched a satellite 'Bhuvan-Climate'
    to track greenhouse gases.
    """
    results = analyzer.analyze_text(test_text, "LIKELY_REAL")
    print(json.dumps(results, indent=2))
