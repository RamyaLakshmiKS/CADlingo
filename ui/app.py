"""
CADlingo Streamlit User Interface

Interactive web application for generating AutoCAD floor plans from text descriptions.

Features:
- Text input for floor plan descriptions
- Real-time code generation
- Visual floor plan preview
- DXF file download
- Sample templates
"""

import streamlit as st
import sys
from pathlib import Path
import os
import io
import base64

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'models'))
sys.path.insert(0, str(project_root / 'src' / 'data'))

from inference import CADGenerator
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="CADlingo - Text to CAD",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_generator():
    """Load the CAD generator model (cached for performance)."""
    try:
        generator = CADGenerator()
        return generator, None
    except Exception as e:
        return None, str(e)


def get_download_link(file_path, link_text):
    """Generate download link for files."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/dxf;base64,{b64}" download="{Path(file_path).name}">{link_text}</a>'
    return href


def main():
    # Header
    st.markdown('<p class="main-header">🏠 CADlingo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Transform natural language into AutoCAD floor plans using AI</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=CADlingo", use_container_width=True)
        
        st.header("About")
        st.info(
            """
            **CADlingo** uses a fine-tuned CodeT5 model to generate AutoCAD code 
            from natural language descriptions of floor plans.
            
            Simply describe your floor plan, and the AI will generate:
            - AutoCAD code
            - Visual preview
            - Downloadable DXF file
            """
        )
        
        st.header("Sample Templates")
        templates = {
            "Small Apartment": "A 800 sq ft floor plan with 1 bedroom, 1 kitchen, and 1 bathroom",
            "Medium House": "A 1500 sq ft floor plan with 3 bedrooms, 1 kitchen, 1 living room, and 2 bathrooms",
            "Large House": "A 2500 sq ft floor plan with 4 bedrooms, 2 bathrooms, 1 kitchen, 1 living room, and 1 dining room",
            "Studio": "A 600 sq ft studio apartment with kitchen and bathroom",
            "Family Home": "A 2000 sq ft floor plan with 3 bedrooms, 2 bathrooms, kitchen, living room, dining room, and storage"
        }
        
        selected_template = st.selectbox(
            "Choose a template:",
            ["Custom"] + list(templates.keys())
        )
        
        if selected_template != "Custom":
            st.session_state.template_text = templates[selected_template]
        
        st.header("Settings")
        num_beams = st.slider("Generation Quality (beams)", 1, 8, 4, help="Higher = better quality but slower")
        temperature = st.slider("Creativity", 0.5, 2.0, 1.0, 0.1, help="Higher = more creative outputs")
        
        st.header("Statistics")
        st.metric("Model", "CodeT5-base")
        st.metric("Training Samples", "1,000+")
        st.metric("Supported Rooms", "10+")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Description")
        
        # Text input
        default_text = st.session_state.get('template_text', "")
        description = st.text_area(
            "Describe your floor plan:",
            value=default_text,
            height=150,
            placeholder="Example: A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room",
            help="Describe the rooms, approximate size, and layout"
        )
        
        # Examples
        with st.expander("Tips for Better Results"):
            st.markdown("""
            - Specify the approximate square footage
            - List the types and numbers of rooms
            - Be clear and concise
            - Use standard room names (bedroom, kitchen, bathroom, living room, etc.)
            
            **Good Examples:**
            - "A 1500 sq ft floor plan with 3 bedrooms, 1 kitchen, and 2 bathrooms"
            - "Small apartment with 1 bedroom, kitchen, bathroom, approximately 800 sq ft"
            - "Large house with 4 bedrooms, 2 bathrooms, kitchen, living room, and dining room"
            """)
        
        # Generate button
        generate_button = st.button("Generate Floor Plan", use_container_width=True)
    
    with col2:
        st.header("Generated Output")
        output_placeholder = st.empty()
        
        with output_placeholder.container():
            st.info("Enter a description and click 'Generate Floor Plan' to start")
    
    # Generation logic
    if generate_button:
        if not description.strip():
            st.error("Please enter a floor plan description")
            return
        
        with st.spinner("Loading AI model..."):
            generator, error = load_generator()
        
        if error:
            st.error(f"Error loading model: {error}")
            st.warning("Please make sure the model is trained. Run the training notebook first.")
            return
        
        with output_placeholder.container():
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Generate code
                status_text.text("Generating AutoCAD code...")
                progress_bar.progress(25)
                
                code = generator.generate_code(
                    description,
                    num_beams=num_beams,
                    temperature=temperature
                )
                
                # Show code
                st.success("Code generated successfully!")
                progress_bar.progress(50)
                
                st.subheader("Generated AutoCAD Code")
                st.code(code, language="lisp")
                
                # Create visualization
                status_text.text("Creating visualization...")
                progress_bar.progress(75)
                
                output_dir = project_root / "data" / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                vis_file = output_dir / "temp_visualization.png"
                dxf_file = output_dir / "generated_floor_plan.dxf"
                
                try:
                    # Generate visualization
                    generator.visualize_floor_plan(code, str(vis_file))
                    
                    st.subheader("Visual Preview")
                    st.image(str(vis_file), use_container_width=True)
                    
                    # Generate DXF
                    status_text.text("Creating DXF file...")
                    generator.code_to_dxf(code, str(dxf_file))
                    progress_bar.progress(100)
                    
                    # Download button
                    st.subheader("Download Files")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        # DXF download
                        with open(dxf_file, 'rb') as f:
                            st.download_button(
                                label="Download DXF File",
                                data=f,
                                file_name="floor_plan.dxf",
                                mime="application/dxf",
                                use_container_width=True
                            )
                    
                    with col_d2:
                        # Code download
                        st.download_button(
                            label="Download Code (TXT)",
                            data=code,
                            file_name="autocad_code.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.markdown(
                        '<div class="success-box"><strong>Success!</strong> Your floor plan has been generated. '
                        'Download the DXF file to open in AutoCAD or any compatible viewer.</div>',
                        unsafe_allow_html=True
                    )
                
                except Exception as viz_error:
                    st.warning(f"Visualization error: {viz_error}")
                    st.info("Code was generated successfully, but visualization failed. You can still download the code.")
                    
                    # Still offer code download
                    st.download_button(
                        label="Download Generated Code",
                        data=code,
                        file_name="autocad_code.txt",
                        mime="text/plain"
                    )
            
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.info("Please try again or use a different description.")
                progress_bar.empty()
                status_text.empty()
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**Project:** CADlingo")
    
    with col_f2:
        st.markdown("**Author:** Ramya Lakshmi KS")
    
    with col_f3:
        st.markdown("**Institution:** University of Florida")


if __name__ == "__main__":
    main()
