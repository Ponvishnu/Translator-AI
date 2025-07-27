# 🌐 Universal AI Translator

> **Powered by Gemini 1.5 Flash + M2M100 | Built with PyTorch, HuggingFace, Streamlit, and LangDetect**

## 📌 Overview

**Universal AI Translator** is an advanced multilingual translation application that delivers highly accurate, culturally aware, and idiom-sensitive translations across major global languages.

Unlike traditional translators, this system combines:

- **Gemini 1.5 Flash** LLM (Google)
- **M2M100** open-source model for fallback
- **Automatic Language Detection**
- A beautifully crafted **Streamlit GUI**
- Optional integration of idiom datasets for fine-tuning

## 🚀 Features

✅ Translate between 12+ languages  
✅ Understands idioms and cultural context  
✅ Auto-detects source language  
✅ Gemini-powered advanced translation  
✅ M2M100 fallback (offline-supported)  
✅ Streamlit GUI for real-time interaction  
✅ Idiom dataset (TSV) support  
✅ 100% Python 3.10+ compatible  
✅ No paid API required (Gemini free-tier supported)

---

## 🧠 Architecture
flowchart TD
    Input[User Input (GUI)]
    Detect[Auto Language Detection (langdetect)]
    Gemini[Primary Translation (Gemini 1.5)]
    M2M[Fallback Translation (M2M100)]
    Output[Translated Output]
    Idioms[Optional Idiom TSV Dataset]

    Input --> Detect
    Detect --> Gemini
    Gemini -->|Failure| M2M
    Gemini --> Output
    M2M --> Output
    Idioms --> Gemini
