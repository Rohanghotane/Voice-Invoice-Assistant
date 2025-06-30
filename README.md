# 🧾 Voice Invoice Assistant

**AI-powered, voice-first invoice assistant** that lets users upload invoices (PDF), query them by voice, and get answers using Retrieval-Augmented Generation (RAG) and modern voice interfaces. Built with `FastAPI`, `LangChain`, `Chroma`, and `Streamlit` – powered by `OpenAI`, `ElevenLabs`, and `DeepGram`.

---

## 🎯 Features

- 📄 **Invoice Ingestion**: Upload PDFs and extract structured data using OCR.
- 🔍 **RAG Pipeline**: Semantic search with LangChain + Chroma + Sentence Transformers.
- 🧠 **LLM Reasoning**: Ask natural language questions about your invoices.
- 🎙️ **Voice Conversation Loop**:
  - Speech-to-text (STT) via Deepgram / ElevenLabs
  - Query LLM with indexed data
  - Text-to-speech (TTS) via ElevenLabs
- 🧑‍💼 **Agent Escalation**: Graceful hand-off when user intent isn't resolved.
- 🌐 **Microservice Architecture**: REST APIs via FastAPI with modular design.
- 🎛️ **Streamlit Frontend**: Clean UI with Start/Stop voice session control.


