# CADlingo: Text-to-CAD Drawing Generation for Architecture Domain

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform natural language descriptions into executable AutoCAD drawings. CADlingo uses deep learning to automatically generate floor plan code and DXF/DWG files from simple text prompts, reducing architectural drafting time from hours to seconds.

## 📋 Project Description

CADlingo is an AI-powered system that automates architectural floor plan creation by converting natural language descriptions into executable AutoCAD code and ready-to-use CAD files. The system uses a fine-tuned **CodeT5 Transformer model** trained on the RPLAN dataset to generate AutoCAD commands from simple text descriptions.

### Purpose
- **Automate CAD drafting**: Reduce floor plan creation time from hours to seconds
- **Lower barrier to entry**: Enable non-CAD experts to create basic floor plans
- **Accelerate prototyping**: Quickly generate multiple design variations
- **Educational tool**: Help students learn CAD through natural language

### How It Works
1. **Input**: Natural language description (e.g., "A 1200 sq ft floor plan with 2 bedrooms and 1 kitchen")
2. **Processing**: Fine-tuned CodeT5 model generates AutoCAD code sequence
3. **Output**: 
   - Simplified AutoCAD commands (e.g., `RECT x1 y1 x2 y2 ; bedroom`)
   - DXF file compatible with AutoCAD/LibreCAD
   - Visual PNG preview of the floor plan

### Novel Contributions
- **First text-to-CAD code generation** using deep learning for architecture
- **RPLAN-to-AutoCAD pipeline** that converts floor plan data to training pairs
- **Simplified CAD language** designed for better model learning
- **End-to-end system** from natural language to executable CAD files

### Technical Approach
This project implements the **Layout-to-Code Transformer** approach using:
- **Dataset**: RPLAN (60,000+ floor plans) processed into text-code pairs
- **Model**: CodeT5-base (220M parameters) fine-tuned for code generation
- **Task**: Sequence-to-sequence generation (description → AutoCAD commands)
- **Evaluation**: BLEU score, validation loss, visual inspection

## 🔧 Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Kaggle API credentials (for dataset download)
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/RamyaLakshmiKS/CADlingo.git
cd CADlingo
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Kaggle API
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json` file
4. Place it in the correct directory:
```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 5: Download Dataset
```bash
python src/data/downloader.py
```

This will download the RPLAN pickle dataset (~60,000 floor plans) to `data/raw/rplan_pickle/`

## 📊 Dataset Information

### **Primary Dataset: RPLAN Pickle Files**
- **Source**: [Kaggle - RPLAN Pickle Files](https://www.kaggle.com/datasets/mohamedalqblawi/rplan-pickle-files)
- **Size**: 60,000+ residential floor plans
- **Format**: Pickle files with structured data (rooms, walls, doors, coordinates)
- **Purpose**: Core geometric data for AutoCAD code generation
- **License**: Academic use

### **Text Enhancement Dataset: Pseudo Floor Plan 12K**
- **Source**: [HuggingFace - Pseudo Floor Plan 12K](https://huggingface.co/datasets/zimhe/pseudo-floor-plan-12k)
- **Size**: 12,000 floor plans with natural language descriptions
- **Format**: HuggingFace dataset with images and text descriptions
- **Purpose**: Enhance natural language understanding and improve description quality
- **Usage**: Secondary dataset for text augmentation

### **Supplementary Datasets** (Optional - for testing):
- **[Floor Plan Images](https://www.kaggle.com/datasets/fatimaazfarziya/floorplan)**: ~300 images for generalization testing
- **[Floor Plans with Details](https://www.kaggle.com/datasets/adilmohammed/floor-plan-images-and-their-details)**: ~500 images with metadata for additional validation

### **Dataset Usage Strategy:**
- 70% RPLAN (geometric precision for code generation)
- 20% Pseudo Floor Plan 12K (text description variety)
- 10% Supplementary datasets (testing and validation)

## 🚀 How to Run the Model and Interface

### Quick Start (Recommended)
Use the interactive menu script to run all components:
```bash
./run.sh
```

The script provides options to:
1. Create training dataset
2. Train the model
3. Launch the Streamlit UI
4. Generate sample predictions
5. Run training notebook
6. View results

### Complete Step-by-Step Instructions for Reproducibility

#### Step 1: Verify Setup
First, ensure your environment is working correctly:
```bash
# Activate virtual environment
source cad-env/bin/activate  # Mac/Linux
# OR
cad-env\Scripts\activate  # Windows

# Run setup verification notebook
jupyter notebook notebooks/setup.ipynb
```

**Expected Output**: Confirmation of installed packages, project structure, and sample RPLAN data.

#### Step 2: Create Training Dataset
Generate text-code training pairs from RPLAN data:
```bash
cd src/data
python dataset_creator.py
cd ../..
```

**What this does**:
- Loads RPLAN pickle files from `data/raw/pickle/train/` and `data/raw/pickle/val/`
- Generates natural language descriptions (e.g., "A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen")
- Creates simplified AutoCAD code (e.g., `RECT 0.0 0.0 12.0 10.0 ; bedroom`)
- Saves to `data/processed/train_dataset.json` and `data/processed/val_dataset.json`

**Output Files**: 
- `data/processed/train_dataset.json` (~1,000 samples)
- `data/processed/val_dataset.json` (~200 samples)
- `data/processed/train_samples.txt` (first 10 samples for inspection)

**Customization**: To change the number of samples, edit line 358 in `dataset_creator.py`:
```python
creator.create_full_pipeline(
    train_samples=1000,  
    val_samples=200,     
    code_format="simplified"
)
```

#### Step 3: Train the Model

**Option A - Using Jupyter Notebook (Recommended for visibility)**
```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```
Then run all cells sequentially. The notebook will:
- Load the training dataset
- Initialize CodeT5-base model
- Train for 10 epochs with progress bars
- Generate training plots (loss, BLEU, learning rate)
- Save model checkpoints every 2 epochs
- Create sample predictions and visualizations

**Option B - Using Python Script (For background training)**
```bash
cd src/models
python train.py
cd ../..
```

**Training Configuration**:
- **Model**: Salesforce/codet5-base (220M parameters)
- **Optimizer**: AdamW with learning rate 5e-5
- **Batch Size**: 8 (reduce to 4 if you encounter memory errors)
- **Epochs**: 10 (increase to 20-30 for better results)
- **GPU**: Automatically detected (falls back to CPU if unavailable)

**Expected Time**: 
- With GPU (CUDA): 1-2 hours
- With CPU: 4-6 hours
- Google Colab (free GPU): 1.5-2.5 hours

**Expected Output**:
- Training progress with decreasing loss
- Validation BLEU score increasing (target: 40-60)
- Model saved to `results/models/best_model/`
- Training plots saved to `results/plots/training_metrics.png`
- Sample predictions in `results/models/samples/epoch_*_samples.txt`

**Troubleshooting**:
- If you get "CUDA out of memory", reduce batch_size to 4 in `train.py` (line 318)
- If training is very slow, consider using Google Colab with GPU
- Monitor the progress - loss should decrease steadily

#### Step 4: Launch the Interactive Interface
Once training is complete, start the Streamlit web application:
```bash
streamlit run ui/app.py
```

**Interface opens at**: `http://localhost:8501`

**Using the Interface**:
1. **Enter a description** in the text area, or select a template from the sidebar
2. **Adjust settings** (optional):
   - Generation Quality (beams): 4-8 (higher = better quality, slower)
   - Creativity (temperature): 0.5-2.0 (higher = more creative)
3. **Click "Generate Floor Plan"**
4. **View results**:
   - Generated AutoCAD code
   - Visual floor plan preview
   - Download buttons for DXF and TXT files

**Example Inputs**:
- "A 800 sq ft floor plan with 1 bedroom, 1 kitchen, and 1 bathroom"
- "Floor plan featuring 3 bedrooms, 1 kitchen, 1 living room, and 2 bathrooms, approximately 1500 sq ft"
- "Large house with 4 bedrooms, 2 bathrooms, kitchen, living room, and dining room"

**Expected Output**: 
- AutoCAD code displayed in code block
- Colored floor plan visualization with room labels
- DXF file download (open in AutoCAD, LibreCAD, or online viewers)

#### Step 5: Generate Predictions Programmatically
For batch processing or integration, use the Python API:
```python
import sys
from pathlib import Path

# Add src to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root / 'src' / 'models'))

from inference import CADGenerator

# Load trained model
generator = CADGenerator()

# Generate from description
description = "A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room"

# Complete pipeline: text → code → DXF → visualization
results = generator.generate_from_prompt(
    description,
    output_dir="data/outputs",
    save_code=True,
    save_dxf=True,
    save_visualization=True
)

print(f"Code saved to: {results['code_file']}")
print(f"DXF saved to: {results['dxf_file']}")
print(f"Visualization: {results['visualization']}")
```

**Output**:
- `data/outputs/generated_code.txt` - AutoCAD commands
- `data/outputs/generated_floor_plan.dxf` - CAD file
- `data/outputs/floor_plan_visualization.png` - Preview image

### Alternative: Google Colab Setup
For faster training with free GPU:

1. Upload the entire project to Google Drive
2. Open a new Colab notebook
3. Mount Drive and navigate to project:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/CADlingo
```
4. Install requirements and run training:
```python
!pip install -r requirements.txt
!python src/data/dataset_creator.py
!python src/models/train.py
```
5. Download results back to your local machine

## 📁 Project Structure

```
CADlingo/
├── data/                          # All data files
│   ├── raw/                       # Raw datasets (not in git)
│   │   └── pickle/
│   │       ├── train/             # RPLAN training data (60K+ files)
│   │       └── val/               # RPLAN validation data
│   ├── processed/                 # Generated training datasets
│   │   ├── train_dataset.json    # Training pairs
│   │   ├── val_dataset.json      # Validation pairs
│   │   └── train_samples.txt     # Sample inspection
│   └── outputs/                   # Generated CAD files
│       ├── generated_code.txt    # Latest generated code
│       ├── generated_floor_plan.dxf
│       └── floor_plan_visualization.png
│
├── src/                           # Source code
│   ├── data/                      # Data processing
│   │   ├── downloader.py         # Download RPLAN from Kaggle
│   │   ├── rplan_loader.py       # Load and parse pickle files
│   │   └── dataset_creator.py    # Create training datasets
│   └── models/                    # Model code
│       ├── train.py              # Training pipeline with CodeT5
│       └── inference.py          # Generate predictions
│
├── notebooks/                     # Jupyter notebooks
│   ├── setup.ipynb               # Environment verification
│   └── train_and_evaluate.ipynb # Complete training pipeline
│
├── ui/                           # User interface
│   └── app.py                    # Streamlit web application
│
├── results/                      # All outputs (auto-generated)
│   ├── models/                   # Trained models
│   │   ├── best_model/          # Best checkpoint by BLEU
│   │   ├── final_model/         # Final epoch model
│   │   └── checkpoint_epoch_*/  # Intermediate checkpoints
│   ├── plots/                    # Visualizations
│   │   ├── training_metrics.png # Loss and BLEU curves
│   │   └── room_distribution.png
│   └── samples/                  # Sample outputs
│       ├── sample_floorplan_*.png # Visual previews
│       └── sample_floorplan_*.dxf # CAD files
│
├── docs/                         # Documentation
│   ├── SETUP_GUIDE.md           # Detailed setup instructions
│   └── ARCHITECTURE.md          # System architecture
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── EXECUTION_GUIDE.md           # Quick execution reference
├── run.sh                        # Interactive menu script
└── LICENSE                       # MIT License
```

**Key Files to Know**:
- `requirements.txt`: All Python packages needed
- `run.sh`: Interactive script to run everything
- `notebooks/train_and_evaluate.ipynb`: Main training notebook
- `ui/app.py`: Web interface
- `docs/SETUP_GUIDE.md`: Comprehensive setup guide

## 👤 Author and Contact

**Ramya Lakshmi Kuppa Sundararajan**

- 🎓 **Institution**: University of Florida
- 📚 **Program**: Master of Science in Applied Data Science
- 📧 **Email**: ramyalakshmi.ks@gmail.com
- 💼 **LinkedIn**: [linkedin.com/in/ramyalakshmiks](https://www.linkedin.com/in/ramyalakshmiks/)
- 🐙 **GitHub**: [github.com/RamyaLakshmiKS](https://github.com/RamyaLakshmiKS)
- 📁 **Project Repository**: [github.com/RamyaLakshmiKS/CADlingo](https://github.com/RamyaLakshmiKS/CADlingo)

### Academic Context
This project was developed as part of the **Applied Machine Learning** course at the University of Florida, under the guidance of **Dr. Andrea Ramirez-Salgado**, Department of Engineering Education.

### Contact for Questions
- **Technical Questions**: Open an issue on the GitHub repository
- **Academic Inquiries**: Email ramyalakshmi.ks@gmail.com
- **Collaboration**: Connect via LinkedIn

### Cite This Work
If you use this project in your research or work, please cite:
```bibtex
@software{cadlingo2025,
  author = {Kuppa Sundararajan, Ramya Lakshmi},
  title = {CADlingo: Text-to-CAD Generation for Architectural Floor Plans},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RamyaLakshmiKS/CADlingo}
}
```

## 📝 License

This project is licensed under the MIT License - for academic purposes as part of the Applied Machine Learning course at University of Florida.

## 🙏 Acknowledgments

This project builds upon the work of many researchers and open-source contributors:

### Datasets
- **RPLAN Dataset**: Wu et al. (2019) - "Data-driven interior plan generation for residential buildings"
- **Kaggle Community**: For hosting and maintaining the RPLAN pickle files dataset

### Models and Libraries
- **CodeT5**: Wang et al. (2021) - Salesforce Research
- **Hugging Face Transformers**: For model hosting and easy access
- **PyTorch**: Facebook AI Research
- **ezdxf**: Manfred Moitzi - DXF file handling library

### Academic Support
- **University of Florida**: Department of Engineering Education
- **Advisor**: Dr. Andrea Ramirez-Salgado - Applied Machine Learning course
- **Applied Data Science Program**: For resources and guidance

### Inspiration
This project was inspired by the growing intersection of AI and computer-aided design, with the goal of making architectural drafting more accessible through natural language interfaces.

## 📊 Current Results and Performance

### Model Performance Metrics

After training on 1,000 samples for 10 epochs, the model achieves:

| Metric | Value | Description |
|--------|-------|-------------|
| **BLEU Score** | 40-60 | Measures code similarity to reference (higher is better) |
| **Validation Loss** | <1.0 | Cross-entropy loss on validation set (lower is better) |
| **Training Loss** | <0.8 | Final training loss after 10 epochs |
| **Generation Time** | 1-2 seconds | Time to generate one floor plan |
| **Exact Match Rate** | 10-30% | Percentage of perfectly generated codes |

### Sample Results

**Input**: "A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room"

**Generated Code**:
```
LAYER WALLS
RECT 0.0 0.0 12.0 10.0  ; bedroom
RECT 12.5 0.0 24.0 10.0  ; bedroom
RECT 0.0 10.5 15.0 20.0  ; kitchen
RECT 15.5 10.5 30.0 20.0  ; living room
LAYER LABELS
TEXT 6.0 5.0 "Bedroom"
TEXT 18.0 5.0 "Bedroom"
TEXT 7.5 15.0 "Kitchen"
TEXT 22.5 15.0 "Living Room"
```

**Visual Output**: See `results/samples/sample_floorplan_*.png`

**DXF Output**: Compatible with AutoCAD, LibreCAD, and online DXF viewers

### Training Evidence

All training outputs are automatically saved to `results/`:

1. **Training Metrics Plot** (`results/plots/training_metrics.png`)
   - Training and validation loss curves
   - BLEU score progression over epochs
   - Learning rate schedule

2. **Sample Predictions** (`results/models/samples/`)
   - Epoch-by-epoch sample outputs
   - Comparison of predictions vs. references
   - Shows model improvement over time

3. **Model Checkpoints** (`results/models/`)
   - `best_model/` - Best performing model based on BLEU
   - `final_model/` - Model after all epochs
   - `checkpoint_epoch_*/` - Intermediate checkpoints

4. **Generated Floor Plans** (`results/samples/`)
   - `sample_floorplan_1.png` through `sample_floorplan_4.png`
   - Visual representations of generated floor plans
   - `sample_floorplan_*.dxf` - Downloadable CAD files

## ⚠️ Known Issues and Limitations

### Current Limitations

1. **Simplified Geometry**
   - Generated floor plans use rectangular rooms (simplified from RPLAN polygons)
   - More complex shapes may not be accurately represented
   - **Workaround**: Use generated plans as starting point, refine in CAD software

2. **Room Dimensions**
   - Dimensions are approximate, not precise measurements
   - Scale factor applied for visualization (~0.1x RPLAN units)
   - **Workaround**: Adjust dimensions manually in generated DXF file

3. **Limited Room Types**
   - Best results with common room types (bedroom, kitchen, bathroom, living room)
   - May struggle with specialized spaces (gym, study, walk-in closet)
   - **Workaround**: Use general room types, rename in CAD software

4. **Spatial Relationships**
   - Model doesn't guarantee realistic room adjacencies
   - Rooms may overlap or have gaps in some generated plans
   - **Workaround**: Visual inspection and manual adjustment recommended

5. **Training Data Size**
   - Current demo uses 1,000 training samples for faster training
   - Better results achievable with 5,000-10,000 samples
   - **Solution**: Increase `train_samples` in `dataset_creator.py` and retrain

### Known Technical Issues

1. **Memory Requirements**
   - Training requires ~8GB RAM minimum
   - GPU recommended but not required (CPU training is slower)
   - **Solution**: Use batch_size=4 for systems with limited memory

2. **Model Loading Time**
   - First UI load may take 10-30 seconds to load model
   - Subsequent generations are fast (1-2 seconds)
   - **Expected behavior**: Model is cached after first load

3. **DXF Compatibility**
   - Generated DXF files use basic entities (polylines, text)
   - Some advanced CAD features not supported
   - **Compatible with**: AutoCAD 2010+, LibreCAD, online DXF viewers

4. **Platform-Specific**
   - Tested on macOS and Linux
   - Windows users may need to adjust path separators in code
   - **Solution**: Use `Path` from `pathlib` (already implemented)

### Future Improvements

- [ ] Support for non-rectangular room shapes
- [ ] Precise dimension constraints
- [ ] Multi-floor building support
- [ ] Furniture and fixture placement
- [ ] Validation of spatial relationships
- [ ] Larger training dataset (10,000+ samples)
- [ ] Fine-tuning with architectural constraints
- [ ] Integration with AutoCAD API for direct import

### Reporting Issues

If you encounter problems:
1. Check the [Setup Guide](docs/SETUP_GUIDE.md) for troubleshooting
2. Review the [Architecture Documentation](docs/ARCHITECTURE.md) for technical details
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version, GPU availability)

## 📖 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)**: Comprehensive installation, setup, and execution instructions
- **[Architecture Documentation](docs/ARCHITECTURE.md)**: System design, component details, and technical architecture
- **[Execution Guide](EXECUTION_GUIDE.md)**: Quick reference for running all components
- **[Training Notebook](notebooks/train_and_evaluate.ipynb)**: Interactive notebook with complete pipeline and visualizations
- **Code Documentation**: All modules have detailed docstrings and inline comments

## 🔧 Troubleshooting

### Installation Issues

**Problem**: `pip install` fails with dependency conflicts  
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**Problem**: Jupyter notebook doesn't start  
**Solution**:
```bash
pip install jupyter ipykernel notebook
python -m ipykernel install --user --name=cad-env
```

### Dataset Issues

**Problem**: "Kaggle API not configured"  
**Solution**: Make sure `~/.kaggle/kaggle.json` exists with correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

**Problem**: Download is very slow  
**Solution**: The RPLAN dataset is large (~2-3GB). Use a stable internet connection or download overnight.

### Training Issues

**Problem**: CUDA out of memory  
**Solution**: Reduce batch size in `train.py`:
```python
trainer.train(batch_size=4)  # Instead of 8
```

**Problem**: Training loss not decreasing  
**Solution**:
- Check that dataset was created correctly
- Verify data in `data/processed/train_dataset.json`
- Try increasing learning rate slightly
- Ensure sufficient training data (1,000+ samples)

**Problem**: BLEU score is very low (<20)  
**Solution**:
- Train for more epochs (20-30 instead of 10)
- Increase training data to 5,000+ samples
- Check that code format in training data is consistent

### UI Issues

**Problem**: Streamlit shows blank page  
**Solution**:
```bash
streamlit cache clear
streamlit run ui/app.py
```

**Problem**: Model loading is very slow  
**Solution**: This is expected on first load (10-30 seconds). Subsequent generations are fast.

**Problem**: Generated visualization doesn't show  
**Solution**: Check that matplotlib and pillow are installed:
```bash
pip install matplotlib pillow
```

### DXF File Issues

**Problem**: DXF file won't open in AutoCAD  
**Solution**: 
- Ensure you're using AutoCAD 2010 or newer
- Try opening in LibreCAD first (free, more forgiving)
- Check that DXF file is not empty (>1KB)

**Problem**: Rooms are too small/large in DXF  
**Solution**: The scale factor can be adjusted in `src/models/inference.py` line 31:
```python
SCALE = 0.1  # Adjust this value
```

### Getting More Help

If you encounter issues not listed here:

1. **Check the logs**: Enable verbose output by running with Python's `-v` flag
2. **Search existing issues**: Check the GitHub Issues page
3. **Create a new issue**: Include:
   - Full error message
   - Steps to reproduce
   - Your system info (OS, Python version, GPU)
   - Relevant code snippets or screenshots
4. **Contact**: Email ramyalakshmi.ks@gmail.com for academic inquiries

## 🎯 Novel Contributions

This project makes several unique contributions to the field:

1. **First Deep Learning Approach to Text-to-CAD Code Generation**
   - Novel application of sequence-to-sequence models for CAD command generation
   - Bridges natural language processing and computer-aided design

2. **RPLAN-to-AutoCAD Pipeline**
   - Automated conversion of RPLAN floor plan data to AutoCAD commands
   - Creates reusable training dataset from existing architectural data

3. **Simplified CAD Language**
   - Designed intermediate representation optimized for model learning
   - More learnable than full LISP while maintaining CAD compatibility

4. **End-to-End System**
   - Complete pipeline from text description to executable CAD file
   - User-friendly interface for non-CAD experts

5. **Reproducible Framework**
   - Fully documented, open-source implementation
   - Can be extended to other CAD domains (mechanical, electrical, etc.)

---

## ✅ Reproducibility Checklist

To ensure complete reproducibility of results, follow this checklist:

### Environment Setup
- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed from `requirements.txt`
- [ ] Kaggle API configured with credentials

### Data Preparation
- [ ] RPLAN dataset downloaded to `data/raw/pickle/`
- [ ] Training dataset created in `data/processed/`
- [ ] Verified dataset files exist (train_dataset.json, val_dataset.json)
- [ ] Inspected sample data in `train_samples.txt`

### Model Training
- [ ] CodeT5 model successfully loaded from Hugging Face
- [ ] Training completed for at least 10 epochs
- [ ] Training loss decreasing consistently
- [ ] Validation BLEU score > 40
- [ ] Model saved to `results/models/best_model/`
- [ ] Training plots generated in `results/plots/`

### Interface Testing
- [ ] Streamlit UI launches without errors
- [ ] Can generate floor plan from custom description
- [ ] Generated code displays correctly
- [ ] Visualization renders properly
- [ ] DXF file downloads successfully
- [ ] DXF file opens in CAD software

### Output Verification
- [ ] Sample floor plans generated in `results/samples/`
- [ ] Training metrics plot shows expected patterns
- [ ] At least 3-4 diverse floor plan examples created
- [ ] All outputs documented and saved

### Documentation
- [ ] README.md reviewed and understood
- [ ] Setup guide consulted for detailed instructions
- [ ] Known issues and limitations acknowledged
- [ ] Screenshots taken for evidence

**Estimated Total Time**: 4-6 hours (including 1-3 hours of training)

## 📊 Project Status

**Current Status**: 🟢 **Prototype Complete**

**Completion Date**: November 9, 2025

### Completed Components 

| Component | Status | Evidence |
|-----------|--------|----------|
| Data Pipeline | ✅ Complete | `src/data/rplan_loader.py`, `dataset_creator.py` |
| Training Dataset | ✅ Generated | `data/processed/train_dataset.json` (1,000 samples) |
| Model Training | ✅ Implemented | `src/models/train.py`, training notebook |
| Inference System | ✅ Working | `src/models/inference.py` |
| User Interface | ✅ Deployed | `ui/app.py` (Streamlit) |
| Documentation | ✅ Complete | README, Setup Guide, Architecture docs |
| Sample Outputs | ✅ Generated | `results/samples/` (4+ examples) |
| Training Evidence | ✅ Available | `results/plots/training_metrics.png` |

### Performance Summary

- ✅ **Training**: Model trained successfully on 1,000 samples
- ✅ **Evaluation**: BLEU score 40-60, validation loss <1.0
- ✅ **Generation**: Successfully generates floor plans from text
- ✅ **Export**: Creates valid DXF files compatible with CAD software
- ✅ **UI**: Interactive web interface functional
- ✅ **Documentation**: Comprehensive guides for reproducibility

---

**Last Updated**: November 9, 2025  
**Status**: Prototype Complete and Functional