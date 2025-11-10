# CADlingo: Complete Setup and Execution Guide

## 📋 Quick Start Guide

This guide walks you through the complete process of setting up and running CADlingo from scratch.

---

## 1. Initial Setup

### Clone the Repository
```bash
git clone https://github.com/RamyaLakshmiKS/CADlingo.git
cd CADlingo
```

### Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv cad-env

# Activate environment
source cad-env/bin/activate  # macOS/Linux
# OR
cad-env\Scripts\activate  # Windows
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Download Dataset

### Setup Kaggle API
1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json` file
4. Place in correct directory:

```bash
# macOS/Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %HOMEPATH%\.kaggle
move %HOMEPATH%\Downloads\kaggle.json %HOMEPATH%\.kaggle\
```

### Download RPLAN Dataset
```bash
python src/data/downloader.py
```

**Expected Output:**
- Downloads to `data/raw/pickle/train/` and `data/raw/pickle/val/`
- ~60,000 floor plan files
- File size: ~2-3 GB

---

## 3. Create Training Dataset

Run the dataset creator to generate text-code pairs:

```bash
cd src/data
python dataset_creator.py
```

**What this does:**
- Loads RPLAN pickle files
- Generates natural language descriptions
- Creates simplified AutoCAD code
- Saves to `data/processed/train_dataset.json` and `val_dataset.json`

**Expected time:** 10-15 minutes for 1000 samples

---

## 4. Train the Model

### Option A: Using Python Script
```bash
cd src/models
python train.py
```

### Option B: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```

Then run all cells in order.

**Training Configuration:**
- Model: CodeT5-base
- Epochs: 10 (increase to 20-30 for better results)
- Batch Size: 8 (adjust based on GPU)
- Expected Time: 1-3 hours

**Output:**
- Trained model saved to `results/models/best_model/`
- Training plots saved to `results/plots/`
- Sample predictions in `results/models/samples/`

---

## 5. Run the UI Application

### Start Streamlit App
```bash
streamlit run ui/app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App:
1. Enter a floor plan description (or select a template)
2. Click "Generate Floor Plan"
3. View generated code and visualization
4. Download DXF file for AutoCAD

---

## 6. Generate Predictions Programmatically

### Python Script
```python
from src.models.inference import CADGenerator

# Load model
generator = CADGenerator()

# Generate from description
description = "A 1200 sq ft floor plan with 2 bedrooms, 1 kitchen, and 1 living room"

results = generator.generate_from_prompt(
    description,
    save_code=True,
    save_dxf=True,
    save_visualization=True
)

print(f"Generated code: {results['code_file']}")
print(f"DXF file: {results['dxf_file']}")
print(f"Visualization: {results['visualization']}")
```

---

## 7. Project Structure

```
CADlingo/
├── data/
│   ├── raw/pickle/          # RPLAN dataset
│   ├── processed/           # Training datasets (JSON)
│   └── outputs/             # Generated files
├── src/
│   ├── data/
│   │   ├── downloader.py    # Download RPLAN
│   │   ├── rplan_loader.py  # Load/parse RPLAN
│   │   └── dataset_creator.py  # Create training data
│   └── models/
│       ├── train.py         # Training pipeline
│       └── inference.py     # Generate predictions
├── notebooks/
│   ├── setup.ipynb          # Initial setup verification
│   └── train_and_evaluate.ipynb  # Complete training notebook
├── ui/
│   └── app.py              # Streamlit application
├── results/
│   ├── models/             # Saved model checkpoints
│   ├── plots/              # Training visualizations
│   └── samples/            # Sample outputs
└── docs/
    └── SETUP_GUIDE.md      # This file
```

---

## 8. Expected Outputs and Evidence

### Training Evidence
- **Training plots**: `results/plots/training_metrics.png`
  - Loss curves
  - BLEU score progression
  - Learning rate schedule

### Sample Outputs
- **Floor plan visualizations**: `results/samples/sample_floorplan_*.png`
- **DXF files**: `results/samples/sample_floorplan_*.dxf`
- **Generated code**: `results/models/samples/epoch_*_samples.txt`

### Model Performance
- **BLEU Score**: ~40-60 (depends on training data size)
- **Validation Loss**: <1.0 after 10 epochs
- **Exact Match Rate**: 10-30%

---

## 9. Troubleshooting

### Issue: "Model not found"
**Solution**: Make sure you've run the training step and `results/models/best_model/` exists

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `train.py` (e.g., from 8 to 4)

### Issue: "No module named 'transformers'"
**Solution**: Reinstall requirements: `pip install -r requirements.txt`

### Issue: "Kaggle API error"
**Solution**: Check that `~/.kaggle/kaggle.json` exists and has correct permissions

### Issue: Streamlit app shows "Model loading error"
**Solution**: Train the model first using the notebook or training script

---

## 10. Evaluation Metrics

### How to Evaluate
Run the evaluation section in `notebooks/train_and_evaluate.ipynb` or:

```python
from src.models.train import CADTrainer

trainer = CADTrainer()
trainer.load_datasets()
val_loss, val_bleu, predictions, references = trainer.evaluate()

print(f"Validation BLEU: {val_bleu:.2f}")
print(f"Validation Loss: {val_loss:.4f}")
```

### Key Metrics
- **BLEU Score**: Measures code similarity to references
- **Validation Loss**: Lower is better
- **Exact Match**: Percentage of perfectly generated codes
- **Visual Quality**: Manual inspection of generated floor plans

---

## 11. Next Steps and Extensions

### For Better Results:
1. **More Training Data**: Increase to 5,000-10,000 samples
2. **Longer Training**: Train for 20-30 epochs
3. **Larger Model**: Use CodeT5-large instead of base
4. **Data Augmentation**: Add variations in descriptions

### Feature Extensions:
1. **Dimensions**: Add support for precise room dimensions
2. **Furniture**: Include furniture placement
3. **Multiple Floors**: Support multi-story buildings
4. **Custom Styles**: Different architectural styles
5. **Interactive Editing**: Allow users to modify generated plans

---

## 12. Screenshots and Demo

### Take Screenshots for Documentation:
1. **Streamlit UI**: Screenshot of the interface with a generated floor plan
2. **Training Notebook**: Screenshot of training progress
3. **Sample Outputs**: Export visualizations from `results/samples/`

### Create Demo Video:
```bash
# Record demo showing:
# 1. Entering description
# 2. Generating floor plan
# 3. Downloading DXF
# 4. Opening in AutoCAD viewer
```

---

## 14. Citation and Credits

**RPLAN Dataset**:
```
@inproceedings{wu2019data,
  title={Data-driven interior plan generation for residential buildings},
  author={Wu, Wenming and Fan, Lubin and Liu, Ligang and Wonka, Peter},
  booktitle={ACM Transactions on Graphics (TOG)},
  year={2019}
}
```

**CodeT5 Model**:
```
@inproceedings{wang2021codet5,
  title={CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation},
  author={Wang, Yue and Wang, Weishi and Joty, Shafiq and Hoi, Steven CH},
  booktitle={EMNLP},
  year={2021}
}
```

---

## 15. Contact and Support

**Author**: Ramya Lakshmi Kuppa Sundararajan  
**Email**: ramyalakshmi.ks@gmail.com  
**Institution**: University of Florida  
**GitHub**: [github.com/RamyaLakshmiKS](https://github.com/RamyaLakshmiKS)

---

**Last Updated**: November 9, 2025  
**Version**: 1.0
