"""
Streamlit UI for AI Medical Prescription Analyzer
"""
import streamlit as st
import pandas as pd
from PIL import Image
import json
from prescription_analyzer import PrescriptionAnalyzer

# Page configuration
st.set_page_config(
    page_title="AI Medical Prescription Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .drug-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the prescription analyzer (cached for performance)"""
    return PrescriptionAnalyzer()

def main():
    st.markdown('<h1 class="main-header">💊 AI Medical Prescription Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    try:
        analyzer = load_analyzer()
    except Exception as e:
        st.error(f"Error loading analyzer: {e}")
        st.info("Please ensure all dependencies are installed and API keys are configured.")
        return
    
    # Sidebar
    st.sidebar.title("📋 Analysis Options")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Prescription Image", "Enter Text Manually"]
    )
    
    # API Key configuration in sidebar
    st.sidebar.markdown("### 🔑 API Configuration")
    api_key_input = st.sidebar.text_input(
        "Enter Gemini API Key:",
        type="password",
        help="Get your API key from Google AI Studio"
    )
    
    if api_key_input:
        analyzer.config.GEMINI_API_KEY = api_key_input
        analyzer.gemini_model = analyzer.config.setup_gemini()
        st.sidebar.success("✅ API Key Updated")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Input</h2>', unsafe_allow_html=True)
        
        prescription_data = None
        
        if input_method == "Upload Prescription Image":
            uploaded_file = st.file_uploader(
                "Choose a prescription image...",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a clear image of the prescription"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Prescription", use_column_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Prescription", type="primary"):
                    with st.spinner("Analyzing prescription..."):
                        prescription_data = analyzer.comprehensive_analysis(uploaded_file, is_image=True)
        
        else:  # Manual text input
            text_input = st.text_area(
                "Enter prescription text:",
                height=200,
                placeholder="Enter the prescription details here..."
            )
            
            if text_input.strip():
                if st.button("🔍 Analyze Text", type="primary"):
                    with st.spinner("Analyzing prescription text..."):
                        prescription_data = analyzer.comprehensive_analysis(text_input, is_image=False)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        if prescription_data:
            # Display extracted text
            with st.expander("📄 Extracted Text", expanded=False):
                st.text_area("Raw Text:", prescription_data['extracted_text'], height=100, disabled=True)
            
            # Display recognized drugs
            st.markdown("### 💊 Recognized Medications")
            if prescription_data['recognized_drugs']:
                for i, drug in enumerate(prescription_data['recognized_drugs'], 1):
                    st.markdown(f'<div class="drug-card"><strong>{i}. {drug.title()}</strong></div>', unsafe_allow_html=True)
            else:
                st.warning("No medications recognized. Please check the image quality or text.")
            
            # Display dosage and frequency
            st.markdown("### 📏 Dosage & Frequency")
            dosage_freq = prescription_data['dosage_frequency']
            
            col_dose, col_freq = st.columns(2)
            with col_dose:
                st.write("**Dosages:**")
                if dosage_freq['dosages']:
                    for dosage in dosage_freq['dosages']:
                        st.write(f"• {dosage}")
                else:
                    st.write("None detected")
            
            with col_freq:
                st.write("**Frequencies:**")
                if dosage_freq['frequencies']:
                    for freq in dosage_freq['frequencies']:
                        st.write(f"• {freq}")
                else:
                    st.write("None detected")
            
            # Medical entities
            if prescription_data['medical_entities']:
                st.markdown("### 🏥 Medical Entities")
                entities_df = pd.DataFrame(prescription_data['medical_entities'])
                st.dataframe(entities_df, use_container_width=True)
            
            # Gemini analysis
            st.markdown("### 🤖 AI Analysis")
            with st.expander("View Detailed AI Analysis", expanded=True):
                if "error" not in prescription_data['gemini_analysis'].lower():
                    st.markdown(prescription_data['gemini_analysis'])
                else:
                    st.error("AI analysis failed. Please check your API key configuration.")
    
    # E-commerce section (full width)
    if prescription_data and prescription_data['recognized_drugs']:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🛒 Order Medications Online</h2>', unsafe_allow_html=True)
        
        # Create tabs for each drug
        drug_tabs = st.tabs([f"💊 {drug.title()}" for drug in prescription_data['recognized_drugs']])
        
        for i, (drug, tab) in enumerate(zip(prescription_data['recognized_drugs'], drug_tabs)):
            with tab:
                st.markdown(f"### Order {drug.title()} from:")
                
                ecommerce_links = prescription_data['ecommerce_links'][drug]
                
                # Create columns for e-commerce sites
                cols = st.columns(len(ecommerce_links))
                
                for j, (link_info, col) in enumerate(zip(ecommerce_links, cols)):
                    with col:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 2px solid #ddd; border-radius: 10px; margin: 0.5rem;">
                            <h4>{link_info['site_name']}</h4>
                            <a href="{link_info['url']}" target="_blank">
                                <button style="background-color: #2E86AB; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">
                                    🛒 Search & Order
                                </button>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for informational purposes only. 
        Always consult with healthcare professionals before taking any medications.</p>
        <p>🔒 Your data is processed locally and not stored on our servers.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()