# CADlingo Architecture Documentation

## System Architecture

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CADlingo System Architecture              │
└─────────────────────────────────────────────────────────────┘

Input: Natural Language Description
    "A 1200 sq ft floor plan with 2 bedrooms and 1 kitchen"
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  1. DATA PROCESSING LAYER                                    │
├──────────────────────────────────────────────────────────────┤
│  • RPLAN Loader: Parse pickle files                          │
│  • Extract: Rooms, walls, doors, coordinates                 │
│  • Dataset Creator: Generate training pairs                  │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  2. MODEL LAYER (CodeT5)                                      │
├──────────────────────────────────────────────────────────────┤
│  • Encoder: Process text description                         │
│  • Decoder: Generate AutoCAD code sequence                   │
│  • Attention: Learn description-code mapping                 │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  3. CODE GENERATION LAYER                                     │
├──────────────────────────────────────────────────────────────┤
│  • Tokenization: Convert text to tokens                      │
│  • Beam Search: Generate best code sequence                  │
│  • Post-processing: Format and validate                      │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  4. EXPORT LAYER                                              │
├──────────────────────────────────────────────────────────────┤
│  • Code Parser: Extract drawing commands                     │
│  • DXF Generator: Create CAD file format                     │
│  • Visualizer: Generate preview image                        │
└──────────────────────────────────────────────────────────────┘
                           ↓
Output: AutoCAD Code + DXF File + Visualization
```

---

## Component Details

### 1. Data Processing Layer

**File**: `src/data/rplan_loader.py`

```python
RPlanLoader
├── load_pickle()           # Load raw RPLAN files
├── parse_floor_plan()      # Extract structured data
├── get_file_list()         # Manage train/val splits
└── get_statistics()        # Compute dataset metrics
```

**File**: `src/data/dataset_creator.py`

```python
CADDatasetCreator
├── AutoCADCodeGenerator
│   ├── generate_code()           # Full LISP code
│   └── generate_simplified_code() # Simplified format
├── DescriptionGenerator
│   └── generate_description()     # Natural language
└── create_dataset()               # Complete pipeline
```

**Data Flow**:
```
RPLAN Pickle → Parse → Extract Rooms → Generate Description
                                    ↘
                                      Generate Code → Save Dataset
```

---

### 2. Model Architecture

**Base Model**: Salesforce CodeT5-base (220M parameters)

```
Input Sequence (Max 128 tokens)
         ↓
   ┌─────────────┐
   │  RoBERTa    │
   │  Tokenizer  │
   └─────────────┘
         ↓
   ┌─────────────┐
   │  Encoder    │
   │  (12 layers)│
   │  Multi-head │
   │  Attention  │
   └─────────────┘
         ↓
   ┌─────────────┐
   │  Decoder    │
   │  (12 layers)│
   │  Cross-     │
   │  Attention  │
   └─────────────┘
         ↓
Output Sequence (Max 512 tokens)
```

**Training Configuration**:
- Optimizer: AdamW
- Learning Rate: 5e-5 with warmup
- Batch Size: 8
- Max Input Length: 128 tokens
- Max Output Length: 512 tokens
- Beam Size: 4

---

### 3. Training Pipeline

**File**: `src/models/train.py`

```python
CADTrainer
├── load_datasets()        # Load JSON datasets
├── train_epoch()          # One training epoch
│   ├── Forward pass
│   ├── Loss computation
│   └── Backpropagation
├── evaluate()             # Validation metrics
│   ├── BLEU score
│   ├── Validation loss
│   └── Sample predictions
└── save_model()           # Checkpoint saving
```

**Training Loop**:
```
For each epoch:
  1. Train on batches
     - Tokenize inputs
     - Forward pass through model
     - Calculate loss
     - Backpropagate gradients
     - Update weights
  
  2. Validate
     - Generate predictions
     - Calculate BLEU
     - Save best model
  
  3. Log metrics
     - Training loss
     - Validation loss
     - BLEU score
```

---

### 4. Inference Pipeline

**File**: `src/models/inference.py`

```python
CADGenerator
├── generate_code()              # Text → Code
│   ├── Tokenize input
│   ├── Generate with beam search
│   └── Decode output
├── parse_code_to_elements()     # Code → Structures
├── code_to_dxf()               # Code → DXF file
└── visualize_floor_plan()      # Code → Image
```

**Generation Process**:
```
Text Input
    ↓
Tokenization
    ↓
Model Inference (Beam Search)
    ↓
Decode Tokens → Code String
    ↓
    ├→ Parse → Visualization
    └→ Parse → DXF File
```

---

### 5. User Interface

**File**: `ui/app.py`

**Streamlit Architecture**:
```
┌────────────────────────────────────┐
│         Streamlit Frontend         │
├────────────────────────────────────┤
│  • Text Input Area                 │
│  • Template Selector               │
│  • Generate Button                 │
│  • Code Display                    │
│  • Visualization Panel             │
│  • Download Buttons                │
└────────────────────────────────────┘
           ↓
┌────────────────────────────────────┐
│         CADGenerator (Cached)      │
├────────────────────────────────────┤
│  • Load model once                 │
│  • Generate on demand              │
│  • Create outputs                  │
└────────────────────────────────────┘
```

---

## Code Format

### Simplified AutoCAD Format

Used for training (easier for model to learn):

```
LAYER WALLS
RECT x1 y1 x2 y2  ; bedroom
RECT x1 y1 x2 y2  ; kitchen
LAYER LABELS
TEXT x y "Bedroom"
TEXT x y "Kitchen"
```

### Full LISP Format

Complete AutoCAD LISP commands:

```lisp
; AutoCAD LISP code for floor plan
(command "LAYER" "N" "WALLS" "C" "7" "WALLS" "")
(command "PLINE" (list 0.0 0.0) (list 10.0 0.0) "C")
(command "TEXT" (list 5.0 5.0) "2.5" "0" "Bedroom")
```

---

## Evaluation Metrics

### BLEU Score
Measures n-gram overlap between generated and reference code:
```
BLEU = BP × exp(Σ log(precision_n))
```

### Validation Loss
Cross-entropy loss on validation set:
```
Loss = -Σ y_true × log(y_pred)
```

### Execution Success Rate
Percentage of generated codes that:
1. Parse correctly
2. Create valid DXF
3. Render without errors

---

## Data Format

### Training Dataset (JSON)
```json
{
  "description": "A 1200 sq ft floor plan with 2 bedrooms and 1 kitchen",
  "code": "LAYER WALLS\nRECT 0.0 0.0 12.0 10.0  ; bedroom\n...",
  "metadata": {
    "source_file": "12345.pkl",
    "num_rooms": 3,
    "room_types": ["bedroom", "kitchen", "bathroom"],
    "dimensions": {"width": 40.5, "height": 35.2}
  }
}
```

### DXF Output
Standard AutoCAD DXF format with:
- Polylines for walls
- Text entities for labels
- Layers for organization

---

## Performance Benchmarks

### Training Time
- **1,000 samples**: ~1-2 hours (GPU)
- **5,000 samples**: ~4-6 hours (GPU)
- **10,000 samples**: ~8-12 hours (GPU)

### Inference Time
- **Single prediction**: ~1-2 seconds
- **Batch of 10**: ~5-10 seconds

### Model Size
- **CodeT5-base**: ~850 MB
- **Fine-tuned model**: ~850 MB
- **Total disk space**: ~2 GB (including datasets)

---

## Extension Points

### 1. Improve Code Generation
- Use larger model (CodeT5-large)
- Add reinforcement learning
- Implement code validation layer

### 2. Enhanced Descriptions
- Add GPT-4 Vision for image descriptions
- Support multi-modal inputs
- Include architectural terminology

### 3. Better Visualization
- 3D rendering
- Interactive editing
- Furniture placement

### 4. Advanced Features
- Multi-floor support
- Dimension constraints
- Style transfer
- Compliance checking

---

## Error Handling

### Common Issues and Solutions

**1. Model Generation Errors**
```python
try:
    code = generator.generate_code(description)
except Exception as e:
    # Fallback to template
    code = get_default_template()
```

**2. DXF Creation Errors**
```python
try:
    generator.code_to_dxf(code, output_file)
except ezdxf.DXFError:
    # Log error, return partial result
    logger.error("Invalid DXF structure")
```

**3. Visualization Errors**
```python
try:
    generator.visualize_floor_plan(code)
except ValueError:
    # Show code only
    display_code_text(code)
```

---

## Testing Strategy

### Unit Tests
```python
def test_rplan_loader():
    loader = RPlanLoader()
    data = loader.load_pickle(test_file)
    assert 'rooms' in data

def test_code_generation():
    generator = CADGenerator()
    code = generator.generate_code("2 bedroom apartment")
    assert "LAYER" in code
```

### Integration Tests
```python
def test_end_to_end():
    # Load data → Train → Generate → Verify
    creator = CADDatasetCreator()
    dataset = creator.create_dataset("train", max_samples=10)
    
    trainer = CADTrainer()
    trainer.train(epochs=1)
    
    generator = CADGenerator()
    code = generator.generate_code(dataset[0]['description'])
    
    assert len(code) > 0
```

---

**Document Version**: 1.0  
**Last Updated**: November 9, 2025  
**Author**: Ramya Lakshmi KS
