"""
Google Colab setup script for AI Medical Prescription Analyzer
Run this script first in Google Colab to install all dependencies and configure the environment
"""

# Cell 1: Install system dependencies and Python packages
def install_dependencies():
    """Install all required dependencies for Google Colab"""
    
    print("🚀 Setting up AI Medical Prescription Analyzer in Google Colab...")
    
    # Install system dependencies for OCR
    import subprocess
    import sys
    
    # Install Tesseract OCR
    print("📦 Installing Tesseract OCR...")
    subprocess.run(['apt', 'update'], check=True, capture_output=True)
    subprocess.run(['apt', 'install', '-y', 'tesseract-ocr'], check=True, capture_output=True)
    subprocess.run(['apt', 'install', '-y', 'libtesseract-dev'], check=True, capture_output=True)
    
    # Install Python packages
    print("🐍 Installing Python dependencies...")
    packages = [
        'streamlit==1.28.1',
        'google-generativeai==0.3.2',
        'transformers==4.36.0',
        'torch==2.1.0',
        'torchvision==0.16.0',
        'Pillow==10.1.0',
        'opencv-python==4.8.1.78',
        'pytesseract==0.3.10',
        'pandas==2.1.3',
        'numpy==1.24.3',
        'requests==2.31.0',
        'beautifulsoup4==4.12.2',
        'scikit-learn==1.3.2',
        'nltk==3.8.1',
        'spacy==3.7.2',
        'python-dotenv==1.0.0',
        'pyngrok==7.0.0'  # For tunneling Streamlit
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                      check=True, capture_output=True)
    
    print("✅ All dependencies installed successfully!")

# Cell 2: Download and setup spaCy model
def setup_spacy():
    """Download spaCy model for NLP processing"""
    import subprocess
    import sys
    
    print("🧠 Setting up spaCy NLP model...")
    try:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], 
                      check=True, capture_output=True)
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Could not download spaCy model: {e}")

# Cell 3: Create project files
def create_project_files():
    """Create all necessary project files in Colab"""
    
    # config.py
    config_content = '''"""
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
        r'\\d+\\s*mg',
        r'\\d+\\s*g',
        r'\\d+\\s*ml',
        r'\\d+\\s*mcg',
        r'\\d+\\s*units?',
        r'\\d+\\s*tablets?',
        r'\\d+\\s*capsules?'
    ]
    
    FREQUENCY_PATTERNS = [
        r'once\\s+daily',
        r'twice\\s+daily',
        r'thrice\\s+daily',
        r'\\d+\\s+times?\\s+daily',
        r'every\\s+\\d+\\s+hours?',
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
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    # prescription_analyzer.py (abbreviated for space)
    analyzer_content = '''"""
Core prescription analysis module - Simplified for Colab
"""
import cv2
import pytesseract
import re
import pandas as pd
import numpy as np
from PIL import Image
import google.generativeai as genai
import nltk
from config import Config

class PrescriptionAnalyzer:
    def __init__(self):
        self.config = Config()
        self.gemini_model = self.config.setup_gemini()
        self.setup_nlp()
        
    def setup_nlp(self):
        """Initialize