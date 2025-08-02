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
from config import Config

class PrescriptionAnalyzer:
    def __init__(self):
        self.config = Config()
        self.gemini_model = self.config.setup_gemini()
        self.setup_nlp()
        
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
    
    def recognize_and_standardize_drugs(self, text):
        """Recognize and standardize drug names"""
        recognized_drugs = []
        text_lower = text.lower()
        
        # Pattern matching for drug names
        drug_patterns = [
            r'\b[A-Za-z]+(?:cillin|mycin|prazole|statin|dipine|formin)\b',
            r'\b(?:tab|cap|syp|inj)[\s\.]*([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
            r'\b([A-Za-z]{3,})\s+\d+\s*(?:mg|g|ml|mcg)\b'
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if len(match) > 2:
                    recognized_drugs.append(match.lower().strip())
        
        # Standardize using common drug database
        standardized_drugs = []
        for drug in recognized_drugs:
            standardized = self.standardize_drug_name(drug)
            if standardized and standardized not in standardized_drugs:
                standardized_drugs.append(standardized)
        
        return standardized_drugs
    
    def standardize_drug_name(self, drug_name):
        """Standardize drug name using database lookup"""
        drug_lower = drug_name.lower().strip()
        
        # Direct match
        if drug_lower in self.config.COMMON_DRUGS:
            return drug_lower
        
        # Check synonyms
        for standard_name, synonyms in self.config.COMMON_DRUGS.items():
            if drug_lower in synonyms:
                return standard_name
        
        # Partial matching for complex names
        for standard_name, synonyms in self.config.COMMON_DRUGS.items():
            if any(syn in drug_lower or drug_lower in syn for syn in [standard_name] + synonyms):
                return standard_name
        
        return drug_name  # Return original if no match found
    
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
    
    def get_ecommerce_links(self, drug_names):
        """Generate e-commerce links for ordering medications"""
        ecommerce_links = {}
        
        for drug in drug_names:
            drug_links = []
            for site in self.config.MEDICAL_ECOMMERCE_SITES:
                search_url = site['search_url'].format(drug.replace(' ', '%20'))
                drug_links.append({
                    'site_name': site['name'],
                    'url': search_url
                })
            ecommerce_links[drug] = drug_links
        
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
        
        # Perform analysis
        results = {
            'extracted_text': extracted_text,
            'recognized_drugs': self.recognize_and_standardize_drugs(extracted_text),
            'dosage_frequency': self.validate_dosage_and_frequency(extracted_text),
            'medical_entities': self.extract_medical_entities(extracted_text),
            'gemini_analysis': self.analyze_with_gemini(extracted_text)
        }
        
        # Add e-commerce links
        results['ecommerce_links'] = self.get_ecommerce_links(results['recognized_drugs'])
        
        return results