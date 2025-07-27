# universal_translator.py
# 🌍 Universal AI Translator | Gemini + M2M100 + Auto Lang Detect

import os
import streamlit as st
from langdetect import detect
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# 🌐 Language dictionary
LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "fr": "French", "de": "German", "es": "Spanish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic"
}
LANG_CODES = {v: k for k, v in LANGUAGES.items()}

# ✅ Gemini API from Streamlit Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("❌ GEMINI_API_KEY not found in secrets. Set it in Streamlit → Settings → Secrets.")
    st.stop()

# 🔮 Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 🔁 Load fallback M2M100 model
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# 🌟 Translation Functions
def gemini_translate(text, target_lang):
    try:
        prompt = f"Translate this to {target_lang}. Preserve meaning, idioms, and culture:\n\n{text}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

def m2m_translate(text, src_lang, tgt_lang):
    try:
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated = m2m_model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        return f"⚠️ Fallback error: {e}"

# 🌍 Streamlit UI
st.set_page_config("Universal AI Translator", page_icon="🌍", layout="centered")

st.markdown("""
# 🌐 Universal AI Translator
**Handles idioms, culture, and more**

Type a sentence in any language and translate to another!
""")

text = st.text_area("✍️ Enter text to translate:", height=120)
target_lang = st.selectbox("🌎 Translate to", list(LANGUAGES.values()), index=list(LANGUAGES.values()).index("Hindi"))

if st.button("🔁 Translate"):
    if not text.strip():
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("🔍 Detecting language..."):
            try:
                detected = detect(text)
                src_name = LANGUAGES.get(detected, "Unknown")
                st.info(f"Detected Source Language: **{src_name} ({detected})**")
            except:
                st.warning("⚠️ Couldn't detect language. Assuming English.")
                detected = "en"

        tgt_code = LANG_CODES[target_lang]
        with st.spinner("🔮 Translating with Gemini..."):
            result = gemini_translate(text, target_lang)

        # Fallback if Gemini fails
        if result.startswith("⚠️"):
            st.warning("Switching to fallback model (M2M100)...")
            result = m2m_translate(text, detected, tgt_code)

        st.success("✅ Translation:")
        st.text_area("🌟 Result:", value=result, height=150)

st.markdown("---")
st.caption("Built with ❤️ using Gemini 1.5, M2M100, Streamlit")
