"""
Core prescription analysis module with OCR, NLP, and drug recognition
"""
import cv2
import pytesseract
import re
import pandas as pd
import numpy as np
from PIL import Image
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModel
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from config import Config
from medicine_verifier import MedicineVerifier, STOP_TOKENS, normalize_name

# Prescribed strength, e.g. "500 mg", "0.5 gm", "5 ml"
DOSAGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|gm|ml|units?|i\.?u\.?)\b", re.IGNORECASE)
# Bare strength used by Indian brands, e.g. "Dolo 650" (but not "x 5 days")
BARE_DOSAGE_RE = re.compile(r"\b(\d{2,4})\b(?!\s*(?:days?|weeks?|months?|times?|hrs?|hours?|%))")
NAME_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
DRUG_STEM_RE = re.compile(
    r"\b[A-Za-z]+(?:cillin|mycin|prazole|statin|dipine|formin|olol|azepam|oxetine|"
    r"profen|floxacin|conazole|tidine|ciclovir|sartan|dronate|gliptin|gliflozin|triptan)\b",
    re.IGNORECASE)
NAME_AFTER_FORM_RE = re.compile(
    r"\b(?:tab|cap|syp|susp|inj|oint|cream)\b\.?\s+([A-Za-z][A-Za-z\-]{2,}(?:\s+[A-Za-z][A-Za-z\-]{2,})?)",
    re.IGNORECASE)
# Words that are never part of a medicine name on a prescription line
LINE_NOISE = STOP_TOKENS | {
    "take", "taken", "use", "used", "apply", "twice", "thrice", "once", "daily",
    "day", "days", "week", "weeks", "month", "months", "after", "before", "food",
    "meals", "meal", "morning", "evening", "night", "empty", "stomach", "each",
    "every", "hour", "hours", "then", "the", "for", "with", "from", "when",
    "needed", "continue", "stop", "review", "follow", "advice", "advised",
    "doctor", "patient", "date", "signature", "diagnosis", "x", "name", "age",
    "sex", "weight", "address", "clinic", "hospital", "mbbs", "reg", "phone",
    "mobile", "sign", "complaint", "history",
}

# Dosing abbreviations commonly written on Indian prescriptions
FREQUENCY_ABBREVIATIONS = ("tds", "tid", "bds", "bd", "od", "hs", "qid", "qds",
                           "sos", "prn", "stat", "qam", "qhs", "qod")
FREQUENCY_ABBREVIATION_RE = re.compile(
    r"\b(" + "|".join(FREQUENCY_ABBREVIATIONS) + r")\b", re.IGNORECASE)
DIRECTIVE_RE = re.compile(
    r"^\s*(?:avoid|do not|don't|not|stop|discontinue|continue|follow|review|"
    r"advised|advice|apply|take|use)\b", re.IGNORECASE)
DOSAGE_FORM_RE = re.compile(
    r"\b(?:tab|tablet|cap|capsule|syp|syrup|susp|suspension|inj|injection|"
    r"oint|ointment|cream|drop|drops|inhaler|respules)\b", re.IGNORECASE)


def build_search_query(name, dosage=""):
    """Clean string for an e-pharmacy search box: official name plus strength."""
    parts = [part for part in (name, dosage) if part]
    query = re.sub(r"[^\w\s\.\-/]", " ", " ".join(parts))
    return re.sub(r"\s+", " ", query).strip()


class PrescriptionAnalyzer:
    def __init__(self):
        self.config = Config()
        self.gemini_model = self.config.setup_gemini()
        self.setup_nlp()
        self.verifier = self._load_verifier()

    def _load_verifier(self):
        """Load the local medicine-name database (verification is optional)."""
        try:
            return MedicineVerifier()
        except Exception as e:
            print(f"Warning: medicine database unavailable: {e}")
            return None
        
    def setup_nlp(self):
        """Initialize NLP components"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize HuggingFace NER pipeline for medical entities
            self.ner_pipeline = pipeline(
                "ner", 
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f"Warning: Could not load advanced NLP models: {e}")
            self.ner_pipeline = None
    
    def extract_text_from_image(self, image_path):
        """Extract text from prescription image using OCR"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text.strip()
        except Exception as e:
            return f"Error in text extraction: {str(e)}"
    
    def extract_text_from_pil_image(self, pil_image):
        """Extract text from PIL Image object"""
        try:
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image.convert('RGB'))
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Convert to grayscale and preprocess
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Extract text
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text.strip()
        except Exception as e:
            return f"Error in text extraction: {str(e)}"

    def extract_medicines(self, text):
        """
        Extract medicines with their dosage and frequency, correcting every
        name against the local database (Levenshtein distance).

        Returns one dict per prescription line, each with the raw string, the
        official name, and a clean search string for e-pharmacy URLs.
        """
        medicines = []
        for line in (text or "").splitlines():
            entry = self._parse_medicine_line(line)
            if entry:
                medicines.append(entry)
        return medicines

    def _parse_medicine_line(self, line):
        """Read one prescription line into a corrected medicine record."""
        text = (line or "").strip()
        if len(text) < 3:
            return None
        dosage = self._extract_dosage(text)
        # Instruction lines ("Avoid alcohol", "Follow up after 5 days")
        if DIRECTIVE_RE.match(text) and not dosage and not DRUG_STEM_RE.search(text):
            return None
        raw_name, verification = self._best_name(text)
        if not raw_name:
            return None
        record = self._correct_name(raw_name, verification)
        if not self._looks_prescribed(text, record, dosage):
            return None
        record.update({
            "raw_line": text,
            "dosage": dosage,
            "frequency": self._extract_frequency(text),
        })
        record["search_query"] = build_search_query(record["name"], dosage)
        return record

    def _looks_prescribed(self, text, record, dosage):
        """Reject prose lines that merely resemble a medicine name."""
        if dosage or DOSAGE_FORM_RE.search(text) or DRUG_STEM_RE.search(text):
            return True
        if record["match_type"] in ("exact", "known-brand"):
            return True
        return record["score"] >= self.verifier.AUTO_CORRECT_SCORE if self.verifier else False

    def _best_name(self, text):
        """Longest, best-scoring name span on the line (Levenshtein cross-check)."""
        tokens = [token for token in NAME_WORD_RE.findall(text) if token.lower() not in LINE_NOISE]
        if not tokens:
            return None, None
        if not self.verifier:
            stem = next((token for token in tokens if DRUG_STEM_RE.match(token)), None)
            return (stem or tokens[0]), None

        best_span, best_verification = None, None
        for start in range(len(tokens)):
            for size in (1, 2, 3, 4):
                span = " ".join(tokens[start:start + size])
                if len(span) < 4:
                    continue
                verification = self.verifier.verify(span)
                if best_verification is None:
                    best_span, best_verification = span, verification
                    continue
                if verification["score"] > best_verification["score"] + 1e-9:
                    best_span, best_verification = span, verification
                elif abs(verification["score"] - best_verification["score"]) < 1e-9 and len(span) > len(best_span):
                    best_span, best_verification = span, verification
        if best_verification and best_verification["score"] >= self.verifier.MIN_SCORE:
            return best_span, best_verification

        # No database match: fall back to prescription heuristics
        stem = next((token for token in tokens if DRUG_STEM_RE.match(token)), None)
        if stem:
            return stem, None
        after_form = NAME_AFTER_FORM_RE.search(text)
        if after_form:
            return after_form.group(1).strip(), None
        if self._extract_dosage(text):
            return tokens[0], None
        return None, None

    @staticmethod
    def _pretty(name):
        """Title-case names stored in all caps or all lower case."""
        if not name:
            return name
        if not (name.isupper() or name.islower()):
            return name
        pretty = name.title()
        return re.sub(r"\b(And|Of|The|With)\b", lambda m: m.group(1).lower(), pretty)

    def _correct_name(self, raw_name, verification=None):
        """Correct one raw name to its official generic or brand name."""
        record = {
            "raw_name": raw_name,
            "name": raw_name,
            "type": "unknown",
            "match_type": "none",
            "score": 0.0,
            "confidence": "low",
            "needs_confirmation": True,
            "candidates": [],
        }
        # Curated brand -> generic pairs win over fuzzy guesses (Dolo, Crocin)
        known_brand = self._known_brand_generic(raw_name, brands_only=True)
        if known_brand:
            record.update({
                "name": known_brand,
                "type": "generic",
                "match_type": "known-brand",
                "score": 0.95,
                "confidence": "high",
                "needs_confirmation": False,
            })
            return record

        if self.verifier:
            verification = verification or self.verifier.verify(raw_name)
            record["candidates"] = [
                dict(candidate, name=self._pretty(candidate["name"]))
                for candidate in verification["candidates"]
            ]
            record["score"] = verification["score"]
            record["match_type"] = verification["match_type"]
            record["confidence"] = verification["confidence"]
            record["needs_confirmation"] = verification["needs_confirmation"]
            if record["candidates"]:
                record["name"] = record["candidates"][0]["name"]
                record["type"] = record["candidates"][0]["type"]
            if verification["auto_corrected"]:
                return record

        # Curated Indian/global brand -> generic map (brands outside the FDA list)
        known = self._known_brand_generic(raw_name)
        if known:
            record.update({
                "name": known,
                "type": "generic",
                "match_type": "known-brand",
                "score": 0.95,
                "confidence": "high",
                "needs_confirmation": False,
            })
        return record

    def _known_brand_generic(self, raw_name, brands_only=False):
        """Look up a brand (or generic) name in the curated brand -> generic map."""
        norm = normalize_name(raw_name)
        if not norm:
            return None
        for generic, brands in self.config.COMMON_DRUGS.items():
            if norm in {normalize_name(brand) for brand in brands}:
                return generic.title()
            if not brands_only and norm == normalize_name(generic):
                return generic.title()
        return None

    def _extract_dosage(self, text):
        """Strength on the line, e.g. "500 mg" or a bare "650"."""
        match = DOSAGE_RE.search(text)
        if match:
            return f"{match.group(1)} {match.group(2).lower()}"
        match = BARE_DOSAGE_RE.search(text)
        return match.group(1) if match else ""

    def _extract_frequency(self, text):
        """First recognised dosing frequency on the line."""
        match = FREQUENCY_ABBREVIATION_RE.search(text)
        if match:
            return match.group(0).upper()
        for pattern in self.config.FREQUENCY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found = match.group(0).strip()
                return found.upper() if len(found) <= 4 else found.title()
        return ""

    def recognize_and_standardize_drugs(self, text):
        """Recognize medicines and return their official (corrected) names."""
        names = self._unique_names(self.extract_medicines(text))
        seen = {name.lower() for name in names}
        # Catch names the line parser missed (e.g. several drugs on one line)
        for match in DRUG_STEM_RE.finditer(text or ""):
            corrected = self.standardize_drug_name(match.group(0))
            if corrected and corrected.lower() not in seen:
                seen.add(corrected.lower())
                names.append(corrected)
        return names

    def standardize_drug_name(self, drug_name):
        """Correct one raw drug string to its official generic or brand name."""
        return self._correct_name((drug_name or "").strip())["name"]

    @staticmethod
    def _unique_names(medicines):
        """Distinct official names, in prescription order."""
        names, seen = [], set()
        for medicine in medicines:
            key = medicine["name"].lower()
            if key not in seen:
                seen.add(key)
                names.append(medicine["name"])
        return names

    def validate_dosage_and_frequency(self, text):
        """Extract and validate dosage and frequency information"""
        dosages = []
        frequencies = []
        
        # Extract dosages
        for pattern in self.config.DOSAGE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dosages.extend(matches)
        
        # Extract frequencies
        for pattern in self.config.FREQUENCY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            frequencies.extend(matches)
        
        return {
            'dosages': list(set(dosages)),
            'frequencies': list(set(frequencies))
        }
    
    def analyze_with_gemini(self, text):
        """Use Google Gemini for advanced prescription analysis"""
        try:
            prompt = f"""
            Analyze this medical prescription text and provide:
            1. List of medications mentioned
            2. Dosages and frequencies
            3. Any potential drug interactions or warnings
            4. Medical conditions mentioned
            5. Doctor's instructions
            
            Prescription text: {text}
            
            Provide a structured analysis in JSON format.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini analysis error: {str(e)}"
    
    def extract_medical_entities(self, text):
        """Extract medical entities using NER"""
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            medical_entities = []
            
            for entity in entities:
                if entity['entity_group'] in ['CHEMICAL', 'DISEASE', 'GENE_OR_GENE_PRODUCT']:
                    medical_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score']
                    })
            
            return medical_entities
        except Exception as e:
            print(f"NER extraction error: {e}")
            return []
    
    def get_ecommerce_links(self, medicines):
        """E-pharmacy search links built from each medicine's clean search string."""
        ecommerce_links = {}
        
        for medicine in medicines:
            query = medicine.get('search_query') if isinstance(medicine, dict) else medicine
            query = (query or '').strip()
            if not query or query in ecommerce_links:
                continue
            encoded = quote(query, safe='')
            ecommerce_links[query] = [
                {
                    'site_name': site['name'],
                    'url': site['search_url'].format(encoded)
                }
                for site in self.config.MEDICAL_ECOMMERCE_SITES
            ]
        
        return ecommerce_links
    
    def comprehensive_analysis(self, text_or_image, is_image=False):
        """Perform comprehensive prescription analysis"""
        # Extract text if image is provided
        if is_image:
            if hasattr(text_or_image, 'read'):
                # File upload object
                pil_image = Image.open(text_or_image)
                extracted_text = self.extract_text_from_pil_image(pil_image)
            else:
                # File path
                extracted_text = self.extract_text_from_image(text_or_image)
        else:
            extracted_text = text_or_image
        
        # Corrected medicines: official name, dosage, clean search string
        medicines = self.extract_medicines(extracted_text)
        
        # Perform analysis
        results = {
            'extracted_text': extracted_text,
            'medicines': medicines,
            'recognized_drugs': self._unique_names(medicines),
            'dosage_frequency': self.validate_dosage_and_frequency(extracted_text),
            'medical_entities': self.extract_medical_entities(extracted_text),
            'gemini_analysis': self.analyze_with_gemini(extracted_text)
        }
        
        # Verification detail per medicine (kept for callers of the old shape)
        results['drug_verification'] = [
            {
                'input': medicine['raw_name'],
                'candidates': medicine['candidates'],
                'confidence': medicine['confidence'],
                'needs_confirmation': medicine['needs_confirmation'],
                'phonetic_match': any(c['match_type'] == 'phonetic' for c in medicine['candidates']),
            }
            for medicine in medicines
        ]

        # Add e-commerce links
        results['ecommerce_links'] = self.get_ecommerce_links(medicines)
        
        return results
