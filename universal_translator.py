# universal_translator_app.py
# 🌍 Universal AI Translator — Gemini 1.5 + M2M100 + Streamlit + Auto & Manual Language Selection

import os
import csv
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import google.generativeai as genai

# === 🔐 Load Gemini API Key ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Please set your GEMINI_API_KEY in a .env file or Streamlit secret.")
    st.stop()

# === 🔮 Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# === 📦 Load M2M100 Fallback Model ===
@st.cache_resource(show_spinner="⏳ Loading M2M100 model...")
def load_m2m100():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

tokenizer, m2m_model = load_m2m100()

# === 🌐 Language Definitions ===
LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic"
}
CODE_TO_NAME = LANGUAGES
NAME_TO_CODE = {v: k for k, v in LANGUAGES.items()}

# === 📘 Load Idioms ===
@st.cache_data
def load_idioms(path="idioms_multilingual.tsv"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            return [f"{a} → {b}" for a, b in reader][:10]
    except:
        return []

# === 🧠 Gemini Translation ===
def gemini_translate(text, target_lang_name):
    prompt = f"Translate the following to {target_lang_name}. Consider idioms and cultural nuances:\n\n{text}"
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini failed: {e}"

# === 🔁 M2M100 Fallback ===
def m2m_translate(text, source_lang, target_lang):
    try:
        tokenizer.src_lang = source_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated = m2m_model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        return f"⚠️ Fallback failed: {e}"

# === 🎨 Streamlit UI ===
st.set_page_config("🌐 Universal AI Translator", page_icon="🌍", layout="wide")
st.title("🌍 Universal AI Translator")
st.caption("Powered by Gemini 1.5 + M2M100 | Handles idioms, culture, and fallback.")

st.markdown("#### ✏️ Enter your sentence below (auto detects source language):")
text_input = st.text_area("Input Text", height=150, placeholder="E.g., रस्सी जल गयी, बल नहीं गया")

# === 🧭 Language Selection ===
col1, col2 = st.columns(2)

# Auto-detect language
auto_src_code = "en"
if text_input.strip():
    try:
        auto_src_code = detect(text_input.strip())
        auto_src_name = CODE_TO_NAME.get(auto_src_code, "English")
    except:
        auto_src_code = "en"
        auto_src_name = "English"
else:
    auto_src_name = "English"

with col1:
    use_auto = st.toggle("🌐 Auto-detect source language", value=True)
    if use_auto:
        st.info(f"Detected Language: **{auto_src_name}** ({auto_src_code})")
        src_code = auto_src_code
    else:
        src_name = st.selectbox("🗣️ Source Language", list(NAME_TO_CODE.keys()), index=list(NAME_TO_CODE).index("English"))
        src_code = NAME_TO_CODE[src_name]

with col2:
    tgt_name = st.selectbox("🌍 Target Language", list(NAME_TO_CODE.keys()), index=list(NAME_TO_CODE).index("Hindi"))
    tgt_code = NAME_TO_CODE[tgt_name]

# === 🔄 Translate Button ===
submit = st.button("🔁 Translate")

if submit and text_input.strip():
    with st.spinner("💬 Translating..."):
        result = gemini_translate(text_input.strip(), tgt_name)

    if "⚠️ Gemini failed" in result:
        with st.spinner("⚠️ Gemini failed. Trying fallback model..."):
            result = m2m_translate(text_input.strip(), src_code, tgt_code)

    st.success("✅ Translated Result:")
    st.markdown(f"**{result}**")

# === 📘 Idioms Display ===
idioms = load_idioms()
if idioms:
    with st.expander("📚 Example Idioms (Multilingual)"):
        for i in idioms:
            st.markdown(f"- {i}")
