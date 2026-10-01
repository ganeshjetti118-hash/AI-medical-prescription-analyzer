# Base44 Development Notes

## Project Overview
AI Medical Prescription Analyzer — a Streamlit app that extracts text from
prescription images (Tesseract OCR), recognizes drugs, validates dosage/frequency,
extracts medical entities (HuggingFace biomedical NER), and provides AI analysis
via Google Gemini. Also generates e-commerce links for ordering medications.

## Stack
- **Framework:** Streamlit 1.28.1 (Python 3.11)
- **OCR:** Tesseract (system binary) + pytesseract
- **NLP:** HuggingFace transformers (d4data/biomedical-ner-all), NLTK, spaCy, scikit-learn
- **AI:** Google Gemini (google-generativeai 0.3.2, model: gemini-pro)
- **Image processing:** OpenCV, Pillow

## Running the App
```bash
docker compose -f docker-compose.base44.yml up -d --build
```
The app is served on port 3000. Health check: `/_stcore/health`.

## Key Files
- `app.py` — Streamlit UI
- `config.py` — Configuration (API keys, drug patterns, e-commerce sites)
- `prescription_analyzer.py` — Core analysis logic (OCR, NER, drug recognition, Gemini)
- `setup_colab.py` — Google Colab setup script (not used in Docker)

## Secrets
- `GEMINI_API_KEY` — Google Gemini API key. Optional for boot (config.py has a default
  placeholder); required for the AI analysis feature. Can also be entered in the app sidebar.

## Notes
- Torch is installed from the CPU-only PyTorch index to keep the image smaller.
- The biomedical NER model (~400MB) downloads on first page load from HuggingFace.
  If it fails, the app continues with NER disabled (graceful degradation).
- `gemini-pro` is an older model name; if Google has deprecated it, the AI analysis
  will return an error but the rest of the app still works.
