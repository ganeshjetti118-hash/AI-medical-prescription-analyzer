"""
Streamlit UI for AI Medical Prescription Analyzer
"""
import streamlit as st
import pandas as pd
from PIL import Image
import json
from prescription_analyzer import PrescriptionAnalyzer, build_search_query

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

MATCH_LABELS = {
    'exact': 'official name',
    'alias': 'equivalent name',
    'known-brand': 'known brand',
    'phonetic': 'same pronunciation',
    'spelling': 'similar spelling',
}

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
    
    # Confirmed medicines, reused by the e-pharmacy section
    confirmed_items = []
    
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
            
            # Display recognized drugs with corrected official names
            st.markdown("### 💊 Recognized Medications")
            medicines = prescription_data.get('medicines') or []
            if medicines:
                for i, medicine in enumerate(medicines, 1):
                    raw_name = medicine['raw_name']
                    official_name = medicine['name']
                    heading_name = raw_name if medicine['needs_confirmation'] else official_name
                    heading = heading_name + (f" · {medicine['dosage']}" if medicine.get('dosage') else "")
                    st.markdown(f'<div class="drug-card"><strong>{i}. {heading}</strong></div>', unsafe_allow_html=True)
                    if medicine.get('frequency'):
                        st.caption(f"Frequency: {medicine['frequency']}")
                    
                    match_label = MATCH_LABELS.get(medicine['match_type'], medicine['match_type'])
                    score_pct = int(round(medicine['score'] * 100))
                    
                    chosen_name = official_name
                    if medicine['match_type'] == 'exact':
                        st.success(f"✅ Official name in medicine database: **{official_name}** ({medicine['type']})")
                    elif not medicine['needs_confirmation']:
                        st.success(f"✅ Corrected **{raw_name}** → **{official_name}** ({match_label}, {score_pct}%)")
                    elif medicine['candidates']:
                        st.warning(f"No confident match for **{raw_name}** — closest names by spelling and pronunciation:")
                        options = [
                            f"{c['name']} · {MATCH_LABELS.get(c['match_type'], c['match_type'])} · {int(round(c['score'] * 100))}%"
                            for c in medicine['candidates']
                        ]
                        keep_raw = f"Keep as written: {raw_name}"
                        choice = st.selectbox(
                            "Confirm this medicine:",
                            options + [keep_raw],
                            index=len(options),
                            key=f"confirm_drug_{i}",
                            help="Database matches help with spelling only; they do not confirm the prescription.",
                        )
                        if choice != keep_raw:
                            chosen_name = medicine['candidates'][options.index(choice)]['name']
                        else:
                            chosen_name = raw_name
                    else:
                        st.error("No match in the local medicine database — please confirm this name.")
                        chosen_name = raw_name
                    
                    search_query = build_search_query(chosen_name, medicine.get('dosage', ''))
                    st.caption(f"Search string for e-pharmacy links: `{search_query}`")
                    
                    if search_query and search_query.lower() not in [item['search_query'].lower() for item in confirmed_items]:
                        confirmed_items.append({
                            'name': chosen_name,
                            'dosage': medicine.get('dosage', ''),
                            'search_query': search_query,
                        })
                st.caption("Database matches are a spelling aid only — not clinical validation.")
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
    if prescription_data and confirmed_items:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🛒 Order Medications Online</h2>', unsafe_allow_html=True)
        
        # Links are built from the corrected, e-pharmacy-ready search strings
        ecommerce_links_by_query = analyzer.get_ecommerce_links(confirmed_items)
        tab_labels = [
            f"💊 {item['name']}" + (f" {item['dosage']}" if item['dosage'] else "")
            for item in confirmed_items
        ]
        drug_tabs = st.tabs(tab_labels)
        
        for item, tab in zip(confirmed_items, drug_tabs):
            with tab:
                st.markdown(f"### Order {item['name']} from:")
                st.caption(f"Search: `{item['search_query']}`")
                
                ecommerce_links = ecommerce_links_by_query.get(item['search_query'], [])
                
                # Create columns for e-commerce sites
                cols = st.columns(len(ecommerce_links)) if ecommerce_links else []
                
                for link_info, col in zip(ecommerce_links, cols):
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
