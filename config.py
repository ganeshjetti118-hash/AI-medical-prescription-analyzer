"""
Configuration module for AI Medical Prescription Analyzer
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys - Set these in your environment or Google Colab secrets
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
    
    # Configure Google Gemini
    @staticmethod
    def setup_gemini():
        genai.configure(api_key=Config.GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-pro')
    
    # Drug validation patterns
    DOSAGE_PATTERNS = [
        r'\d+\s*mg',
        r'\d+\s*g',
        r'\d+\s*ml',
        r'\d+\s*mcg',
        r'\d+\s*units?',
        r'\d+\s*tablets?',
        r'\d+\s*capsules?'
    ]
    
    FREQUENCY_PATTERNS = [
        r'once\s+daily',
        r'twice\s+daily',
        r'thrice\s+daily',
        r'\d+\s+times?\s+daily',
        r'every\s+\d+\s+hours?',
        r'bid',
        r'tid',
        r'qid',
        r'od',
        r'bd',
        r'morning',
        r'evening',
        r'night'
    ]
    
    # E-commerce medical websites
    MEDICAL_ECOMMERCE_SITES = [
        {
            'name': '1mg',
            'url': 'https://www.1mg.com',
            'search_url': 'https://www.1mg.com/search/all?name={}'
        },
        {
            'name': 'Netmeds',
            'url': 'https://www.netmeds.com',
            'search_url': 'https://www.netmeds.com/catalogsearch/result/{}/all'
        },
        {
            'name': 'PharmEasy',
            'url': 'https://pharmeasy.in',
            'search_url': 'https://pharmeasy.in/search/all?name={}'
        },
        {
            'name': 'Apollo Pharmacy',
            'url': 'https://www.apollopharmacy.in',
            'search_url': 'https://www.apollopharmacy.in/search-medicines/{}'
        }
    ]
    
    # Common drug database (simplified)
    COMMON_DRUGS = {
        'paracetamol': ['acetaminophen', 'tylenol', 'crocin', 'dolo'],
        'ibuprofen': ['brufen', 'advil', 'combiflam'],
        'amoxicillin': ['augmentin', 'amoxil'],
        'metformin': ['glucophage', 'glycomet'],
        'aspirin': ['disprin', 'ecosprin'],
        'omeprazole': ['prilosec', 'omez'],
        'atorvastatin': ['lipitor', 'atorlip'],
        'amlodipine': ['norvasc', 'amlovas']
    }