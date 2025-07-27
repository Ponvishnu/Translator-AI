# universal_translator_app.py
# 🌍 Universal AI Translator — Gemini 1.5 + M2M100 + Streamlit + Auto Language Detection

import os
import csv
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import google.generativeai as genai

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Please set your GEMINI_API_KEY in a .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# Load M2M100 fallback model
@st.cache_resource(show_spinner="⏳ Loading model...")
def load_m2m100():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

tokenizer, m2m_model = load_m2m100()

# Supported languages
LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic"
}
LANG_NAMES = list(LANGUAGES.values())

# Load idioms
@st.cache_data
def load_idioms(path="idioms_multilingual.tsv"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            return [f"{a} → {b}" for a, b in reader][:10]
    except:
        return []

# Gemini Translation
def gemini_translate(text, target_lang_name):
    prompt = f"Translate the following text to {target_lang_name}. Handle idioms and cultural context:\n\n{text}"
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini failed: {e}"

# Fallback Translation
def m2m_translate(text, source_lang, target_lang):
    try:
        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated = m2m_model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        return f"⚠️ Fallback failed: {e}"

# Streamlit UI
st.set_page_config("🌐 Universal AI Translator", layout="wide", page_icon="🌍")
st.title("🌍 Universal AI Translator")
st.markdown("✨ Understands idioms, culture, and more.")

text_input = st.text_area("✏️ Enter your text (auto-detects language)", height=150, placeholder="Type something...")

if text_input.strip():
    try:
        detected_code = detect(text_input)
        detected_lang = LANGUAGES.get(detected_code, "Unknown")
        st.info(f"🌐 Detected language: **{detected_lang}** ({detected_code})")
    except:
        detected_code = "en"
        st.warning("⚠️ Language detection failed. Defaulting to English.")

    src_lang_code = detected_code if detected_code in LANGUAGES else "en"
    src_lang_name = LANGUAGES.get(src_lang_code, "English")
else:
    src_lang_code = "en"
    src_lang_name = "English"

tgt_lang_name = st.selectbox("🌍 Target Language", LANG_NAMES, index=LANG_NAMES.index("Hindi"))
submit = st.button("🔁 Translate Now")

if submit and text_input.strip():
    tgt_lang_code = [k for k, v in LANGUAGES.items() if v == tgt_lang_name][0]

    with st.spinner("🧠 Translating..."):
        result = gemini_translate(text_input.strip(), tgt_lang_name)

    if "⚠️ Translator AI failed" in result:
        with st.spinner("⚠️ Translator AI failed. Using fallback..."):
            result = m2m_translate(text_input.strip(), src_lang_code, tgt_lang_code)

    st.success("✅ Translated Output")
    st.markdown(f"**{result}**")

# Sample idioms
idioms = load_idioms()
if idioms:
    with st.expander("📚 Sample Idioms"):
        for i in idioms:
            st.markdown(f"- {i}")
