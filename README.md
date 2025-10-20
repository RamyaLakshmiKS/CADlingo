# CADlingo: Text-to-CAD Drawing Generation for Architecture Domain

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform natural language descriptions into executable AutoCAD drawings. CADlingo uses deep learning to automatically generate floor plan code and DXF/DWG files from simple text prompts, reducing architectural drafting time from hours to seconds.

## 📋 Project Description

CADlingo automates architectural floor plan creation by converting natural language descriptions into executable AutoCAD code and ready-to-use CAD files. Using a fine-tuned Transformer model (CodeT5) trained on the RPLAN dataset, the system learns to generate precise AutoCAD LISP commands from textual descriptions of room layouts, walls and doors.

**Input**: _"Draw a 3-bedroom apartment with open kitchen and living room, 1200 sq ft"_  
**Output**: 
- AutoCAD LISP code for drawing commands
- DXF/DWG files ready to open in AutoCAD
- Visual floor plan preview

**Novel Contribution**: First application of deep learning for structured architectural layout-to-code generation, enabling architects to generate base drawings in seconds rather than hours.

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

## 🚀 How to Run

### Explore the Dataset
```bash
jupyter notebook notebooks/setup.ipynb
```

### Generate Training Data (when datasets are downloaded)
```bash
python src/data/dataset_creator.py
```

### Train the Model (Coming Soon)
```bash
python src/models/train.py
```

### Run the Full Pipeline (Coming Soon)
```bash
streamlit run app.py
```

## 📁 Project Structure

```
CADlingo/
├── data/
│   ├── raw/                    # Raw datasets 
│   ├── processed/              # Train/val/test splits
│   └── outputs/                # Generated CAD files (intermediate)
├── src/
│   ├── data/
│   │   ├── downloader.py       # Dataset download script
│   │   ├── rplan_loader.py     # Load and process pickle files
│   │   └── dataset_creator.py  # Generate training data
│   └── models/
│       ├── train.py            # Model training pipeline
│       └── inference.py        # Generate AutoCAD code
├── notebooks/
│   └── setup.ipynb             # Setup & environment verification
├── results/
│   ├── plots/                  # Exploratory figures/visualizations
│   └── samples/                # Sample outputs (LISP/DXF/PNG)
├── ui/
│   └── app.py                  # Streamlit app (coming soon)
├── docs/
│   └── SETUP_GUIDE.md          # Quick setup guide (coming soon)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 👤 Author

**Ramya Lakshmi Kuppa Sundararajan**
- Institution: University of Florida
- Program: Master's in Applied Data Science
- Email: ramyalakshmi.ks@gmail.com
- LinkedIn: [linkedin.com/ramyalakshmiks](https://www.linkedin.com/in/ramyalakshmiks/)
- GitHub: [github.com/RamyaLakshmiKS](https://github.com/RamyaLakshmiKS)

## 📝 License

This project is licensed under the MIT License - for academic purposes as part of the Applied Machine Learning course at University of Florida.

## 🙏 Acknowledgments

- RPLAN dataset creators
- Hugging Face for CodeT5 model
- University of Florida, Department of Engineering Education
- Advisor: Dr. Andrea Ramirez-Salgado

---

**Last Updated**: October 19, 2025  
**Project Status**: 🟢 Week 1 of 6