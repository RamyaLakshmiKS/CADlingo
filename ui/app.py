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
import matplotlib.patches as mpatches
import numpy as np

# Add improvements module
sys.path.insert(0, str(project_root / 'src' / 'improvements'))
from improvement_modules import GeometricValidator, AutomatedEvaluator


# Room type colors for better visualization
ROOM_COLORS = {
    'living_room': '#FFB6C1',      # Light pink
    'bedroom': '#87CEEB',          # Sky blue
    'master_bedroom': '#4682B4',   # Steel blue
    'child_bedroom': '#ADD8E6',    # Light blue
    'guest_bedroom': '#B0E0E6',    # Powder blue
    'kitchen': '#FFD700',          # Gold
    'dining_room': '#FFA07A',      # Light salmon
    'bathroom': '#98FB98',         # Pale green
    'balcony': '#F0E68C',          # Khaki
    'storage': '#D3D3D3',          # Light gray
    'hallway': '#DDA0DD',          # Plum
    'laundry': '#F5DEB3',          # Wheat
    'office': '#E6E6FA',           # Lavender
    'default': '#DCDCDC'           # Gainsboro
}


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


def create_enhanced_visualization(code, output_file, figsize=(12, 10)):
    """Create enhanced visualization with color-coded rooms and scale legend."""
    import re
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Parse rooms from code
    rect_pattern = r'RECT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"([^"]+)"'
    rooms = re.findall(rect_pattern, code)
    
    if not rooms:
        ax.text(0.5, 0.5, 'No valid rooms found', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate bounds
    all_x = []
    all_y = []
    for x, y, w, h, _ in rooms:
        x, y, w, h = float(x), float(y), float(w), float(h)
        all_x.extend([x, x + w])
        all_y.extend([y, y + h])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Add padding
    padding = 2
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    # Draw rooms with color coding
    legend_elements = []
    room_types_seen = set()
    
    for x, y, w, h, room_type in rooms:
        x, y, w, h = float(x), float(y), float(w), float(h)
        
        # Get color for room type
        color = ROOM_COLORS.get(room_type, ROOM_COLORS['default'])
        
        # Draw rectangle
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add room label
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Format room name
        display_name = room_type.replace('_', ' ').title()
        area = w * h
        
        ax.text(
            center_x, center_y,
            f'{display_name}\n{area:.1f}m²',
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        # Add to legend if not seen
        if room_type not in room_types_seen:
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', label=display_name)
            )
            room_types_seen.add(room_type)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add scale reference
    scale_length = 5  # 5 meters
    scale_x = min_x - padding + 1
    scale_y = min_y - padding + 0.5
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', linewidth=3)
    ax.plot([scale_x, scale_x], [scale_y - 0.2, scale_y + 0.2], 'k-', linewidth=2)
    ax.plot([scale_x + scale_length, scale_x + scale_length], [scale_y - 0.2, scale_y + 0.2], 'k-', linewidth=2)
    ax.text(scale_x + scale_length/2, scale_y - 0.7, f'{scale_length}m', ha='center', fontsize=10, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Width (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (meters)', fontsize=12, fontweight='bold')
    ax.set_title('Generated Floor Plan Layout', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
            title='Room Types',
            title_fontsize=11,
            framealpha=0.9
        )
    
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Header
    st.markdown('<p class="main-header">🏠 CADlingo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Transform natural language into AutoCAD floor plans using AI</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=CADlingo", width=300)
        
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
        generate_button = st.button("Generate Floor Plan")
    
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
                    # Generate enhanced visualization
                    create_enhanced_visualization(code, str(vis_file))
                    
                    st.subheader("Visual Preview")
                    st.image(str(vis_file))
                    
                    # Add validation report
                    st.subheader("Quality Assessment")
                    validator = GeometricValidator()
                    
                    # Parse rooms from code for validation
                    import re
                    rect_pattern = r'RECT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"([^"]+)"'
                    room_matches = re.findall(rect_pattern, code)
                    
                    parsed_rooms = []
                    for x, y, w, h, room_type in room_matches:
                        x, y, w, h = float(x), float(y), float(w), float(h)
                        parsed_rooms.append({
                            'type': room_type,
                            'center': (x + w/2, y + h/2),
                            'width': w,
                            'height': h,
                            'area': w * h
                        })
                    
                    if parsed_rooms:
                        validation_report = validator.validate_floor_plan(parsed_rooms)
                        
                        col_q1, col_q2, col_q3 = st.columns(3)
                        
                        with col_q1:
                            st.metric(
                                "Overall Quality",
                                f"{validation_report['overall_score']:.1f}%",
                                delta="Valid" if validation_report['is_valid'] else "Needs Review"
                            )
                        
                        with col_q2:
                            st.metric(
                                "Room Count",
                                validation_report['total_rooms'],
                                delta=f"{validation_report['valid_rooms']} valid"
                            )
                        
                        with col_q3:
                            overlap_status = "✓ No overlaps" if validation_report['overlap_count'] == 0 else f"⚠ {validation_report['overlap_count']} overlaps"
                            st.metric(
                                "Layout Check",
                                overlap_status
                            )
                        
                        if validation_report['issues']:
                            with st.expander("View Validation Issues"):
                                for issue in validation_report['issues']:
                                    st.warning(issue)
                    
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
                                mime="application/dxf"
                            )
                    
                    with col_d2:
                        # Code download
                        st.download_button(
                            label="Download Code (TXT)",
                            data=code,
                            file_name="autocad_code.txt",
                            mime="text/plain"
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
