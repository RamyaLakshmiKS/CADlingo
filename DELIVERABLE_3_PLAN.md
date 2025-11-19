# **DELIVERABLE 3 EXECUTION PLAN**
### **CADlingo: Production-Grade Refinement**

**Timeline:** Immediate (completion by week before deadline)  
**Primary Goal:** Address professor feedback systematically with measurable improvements and evidence  
**Success Criteria:** Full GitHub evidence + 5-6 page IEEE report + working demo + research poster

---

## **PHASE BREAKDOWN & PRIORITIES**

### **PRIORITY 1: HIGH-IMPACT, QUICK WINS (Days 1-2)**
*These directly address professor's top feedback items*

#### **1.1 Geometric Validation Layer** ✅ **DONE**
- **Status:** `geometric_validator.py` created with full implementation
- **Deliverables:**
  - `GeometricValidator` class: overlap detection, room size validation, adjacency checking
  - `GeometricMetrics` class: IoU, room count accuracy, type accuracy, plausibility scoring
  - Integrated into `inference.py` via `generate_with_validation()` method
  - Captures validation report alongside code generation
  
- **Evidence for Report:**
  - Show before/after validation reports for 5-10 sample outputs
  - Include validation score distributions across test set
  - Demonstrate how validation fixes common issues

#### **1.2 Comprehensive Evaluation Notebook** ✅ **CREATED**
- **File:** `notebooks/evaluation_enhanced.ipynb`
- **Sections:**
  1. GPU setup & environment (HiPerGator B200)
  2. Data augmentation & curriculum learning
  3. Geometric validation implementation
  4. Enhanced metrics (IoU, room count, adjacency, plausibility)
  5. Fine-tuning execution (multi-phase approach)
  6. Model comparison & ensemble methods
  7. Production deployment preparation

- **Target Outputs:**
  - Geometric metrics computed on 50+ test cases
  - Side-by-side examples (description → code → validation report → visualization)
  - Error categorization: classification of failure modes
  - Improvement trajectories showing before/after

**Action Items:**
- [ ] Run geometric validator notebook on test set (Day 2)
- [ ] Generate 20 side-by-side examples for report (Day 2)
- [ ] Create error analysis table (Day 2)

---

### **PRIORITY 2: MODEL IMPROVEMENTS (Days 2-4)**
*Demonstrates technical depth and measurable BLEU improvement*

#### **2.1 Data Augmentation Pipeline** 
- **Goal:** Expand from 1K → 2.5K samples with higher quality and diversity
- **Techniques:**
  - Synonym replacement in descriptions (room names, dimensions, styles)
  - Code obfuscation: variable renaming (RECT1 → LIVING_RECT, etc.)
  - Back-translation: Code→Description→CodeT5→Code for validation
  - Synthetic combinations: mix room configurations from different plans

- **Implementation:**
  - Update `src/data/dataset_creator.py` with augmentation methods
  - Create `src/data/augmentation.py` for standalone augmentation utilities
  - Log augmentation statistics: original count, augmented count, diversity metrics

- **Expected Impact:** BLEU +2-3 from data diversity

- **Action Items:**
  - [ ] Implement augmentation methods (Day 2)
  - [ ] Generate augmented dataset (Day 2-3)
  - [ ] Document augmentation strategy and statistics (Day 3)

#### **2.2 Hyperparameter Optimization on GPU**
- **Goal:** Find optimal configuration using HiPerGator B200
- **Strategy:**
  - Test learning rates: [1e-5, 3e-5, 5e-5, 1e-4]
  - Test batch sizes: [4, 8, 16]
  - Test warmup steps: [100, 500, 1000]
  - Pick top configuration based on validation BLEU + loss curves

- **Timeline:** Day 3 (parallel runs on GPU)
- **Output:** Hyperparameter comparison table with curves

- **Expected Impact:** BLEU +2-4 from optimization

- **Action Items:**
  - [ ] Set up HiPerGator job submission scripts (Day 2 evening)
  - [ ] Run parallel hyperparameter searches (Day 3)
  - [ ] Collect results and create comparison visualization (Day 3)

#### **2.3 Advanced Training Techniques**
- **Apply in final training run (Day 4):**
  - Learning rate warmup (1000 steps)
  - Label smoothing (α=0.1)
  - Early stopping with patience
  - Mixed precision training (fp16) for GPU efficiency

- **Configuration:**
  - Update `train.py` with these techniques
  - Add convergence tracking: loss, BLEU, validation metrics

- **Expected Impact:** BLEU +2-3 from training refinements

- **Expected Outcome by Day 4:**
  - **Baseline (Deliverable 2):** BLEU 23.1
  - **Target (Deliverable 3):** BLEU 27-30
  - **Stretch Goal:** BLEU 32-35

- **Action Items:**
  - [ ] Implement advanced training in `train.py` (Day 3)
  - [ ] Run final training with optimized hyperparameters (Day 4)
  - [ ] Save best checkpoints and training curves (Day 4)

---

### **PRIORITY 3: DOCUMENTATION & VISUALS (Days 4-5)**
*Creates evidence of improvements for report and demo*

#### **3.1 Expanded IEEE Report (5-6 pages)**

**Section 1: Project Summary & Updates** (0.75 page)
- Brief recap of Deliverable 2
- Key improvements in Deliverable 3
- Why these refinements matter

**Section 2: System Architecture with Visuals** (1 page)
- Updated architecture diagram showing validation layer
- Pipeline flowchart: Data → Model → Validation → Output
- Explain evolution from D2 → D3

**Section 3: Refinements Made** (1.25 pages)
- Data augmentation strategy with statistics table
- Hyperparameter tuning results (learning rates, batch sizes)
- Advanced training techniques explained
- Geometric validation layer design
- Include 2-3 figures: augmentation examples, hyperparam curves, validation scores

**Section 4: Enhanced Evaluation & Metrics** (1.25 pages)
- **Comparison Table:** Baseline (D2) vs. Improved (D3)
  - BLEU score, Validation loss, Training time
  - IoU, Room count accuracy, Type accuracy, Adjacency accuracy
  - Layout plausibility, Validation failure rate
- Before/after visualizations
- Error categorization chart

**Section 5: Side-by-Side Examples** (0.75 page)
- 2-3 examples: Description → Code → Validation Report → Visualization
- Include commentary: "What the model got right/wrong"
- Show how validation layer fixed issues

**Section 6: UI Improvements & Usability** (0.5 page)
- Screenshots showing color-coded rooms, scale legend
- Parameter controls (beam size, temperature, validation toggle)
- Example gallery

**Section 7: Limitations & Future Improvements** (0.5 page)
- Visual table: Current challenge → Proposed solution → Timeline
- Discuss architectural feasibility

**Section 8: Responsible AI Reflection** (0.5 page)
- Address fairness in architectural generation
- Bias mitigation for different building types
- Accessibility considerations

- **Deliverables:**
  - [ ] Write all sections with embedded figures (Day 4-5)
  - [ ] Create final PDF (Day 5)
  - [ ] Proofread and refine (Day 5)

#### **3.2 Enhanced UI with Screenshots**
**Streamlit Improvements (Time: 2-3 hours)**

Add to `ui/app.py`:
- **Color-coded rooms** by type (living_room=salmon, bedroom=blue, kitchen=gold, etc.)
- **Scale legend** showing 10ft/5m grid references
- **Example gallery:** 5-10 preset templates users can select
- **Parameter controls:** 
  - Beam size slider (2-10)
  - Temperature slider (0.5-1.5)
  - Toggle: "Enable Geometric Validation"
- **Real-time feedback:** Shows validation report before DXF export
- **Batch processing:** Upload multiple descriptions, generate multiple plans

- **Documentation:**
  - [ ] Implement UI improvements (Day 3-4)
  - [ ] Take screenshots of new features (Day 4)
  - [ ] Include in report (Day 4)

#### **3.3 Updated README**
- **Add Visual Overview:**
  - ASCII architecture diagram at top
  - Performance comparison table (D2 vs D3)
  - Feature checklist
  
- **Include:**
  - HiPerGator GPU setup instructions
  - Quick start guide (3 steps to generate floor plans)
  - Known issues and workarounds
  - Contact information

- **Action Items:**
  - [ ] Create visual overview (Day 4)
  - [ ] Write GPU setup section (Day 4)
  - [ ] Update performance table with new metrics (Day 5)

---

### **PRIORITY 4: RESEARCH POSTER (Day 5)**
*Professional visual summary for presentation*

**Poster Layout (A3 or 36×48"):**

1. **Header** (Top 15%)
   - Title: "CADlingo: Automated Floor Plan Generation from Natural Language"
   - Your name + University of Florida
   - QR code linking to GitHub demo

2. **Abstract & Problem** (15%)
   - 2-3 sentence problem statement
   - Why this matters for architecture/CAD industry

3. **Methodology** (20%)
   - System architecture diagram (small)
   - RPLAN → CodeT5 → DXF pipeline visualization
   - Include data augmentation and validation layer

4. **Results** (25%)
   - **Before/After Metrics Table:**
     - BLEU: 23.1 → 28-30+
     - Val Loss: 2.1 → <1.5
     - Validation failure rate: X% → Y%
   - 2-3 example floor plans (small visualizations)
   - Performance comparison charts

5. **Key Takeaways** (10%)
   - What worked (augmentation, validation, GPU optimization)
   - What didn't (mention challenges honestly)
   - Impact on real-world usage

6. **References & Acknowledgements** (5%)
   - Key papers (CodeT5, RPLAN, etc.)
   - Tools used (PyTorch, HuggingFace, etc.)

- **Action Items:**
  - [ ] Design in Canva or PowerPoint (Day 5)
  - [ ] Export as high-res PDF (1200 DPI) (Day 5)

---

## **GitHub COMMIT EVIDENCE**

**Minimum 3-5 commits showing progression:**

1. **Commit 1:** "feat: Add geometric validation layer"
   - Add `src/models/geometric_validator.py`
   - Integrate into `inference.py`
   - Files changed: 2

2. **Commit 2:** "feat: Implement data augmentation pipeline"
   - Add `src/data/augmentation.py`
   - Update `dataset_creator.py`
   - Add augmentation tests

3. **Commit 3:** "feat: Enhanced evaluation metrics and comprehensive notebook"
   - Add `notebooks/evaluation_enhanced.ipynb`
   - Create `src/evaluation/metrics.py` with IoU, room count, etc.
   - Files changed: 3

4. **Commit 4:** "refactor: Update Streamlit UI with color-coding and validation"
   - Update `ui/app.py` with new features
   - Add color mapping, parameter controls
   - Include screenshots in `docs/UI_IMPROVEMENTS.md`

5. **Commit 5:** "docs: Expand README and report for production readiness"
   - Update `README.md` with visual overview
   - Add HiPerGator setup instructions
   - Create `DELIVERABLE_3_SUMMARY.md`
   - Add final report PDF to `docs/`

---

## **EXPECTED OUTCOMES BY SUBMISSION DATE**

### **GitHub Repository**
- ✅ 5+ commits showing clear progression
- ✅ Updated `requirements.txt` (if new dependencies added)
- ✅ New files: `geometric_validator.py`, `evaluation_enhanced.ipynb`, updated `inference.py`, updated `ui/app.py`
- ✅ Evidence of GPU training (training logs, metrics curves)
- ✅ `DELIVERABLE_3_SUMMARY.md` explaining all changes

### **IEEE Report (PDF)**
- ✅ 5-6 pages (not 4 or fewer like D2)
- ✅ System architecture diagram with validation layer
- ✅ Pipeline flowchart showing RPLAN→CodeT5→Validation→DXF
- ✅ 3+ side-by-side examples with commentary
- ✅ Before/after metrics comparison table
- ✅ Error categorization chart
- ✅ UI screenshots
- ✅ Limitations & future improvements table
- ✅ Responsible AI reflection section
- ✅ Reproducibility: GPU setup, hyperparameters, random seeds documented

### **Updated README**
- ✅ Visual architecture overview at top
- ✅ Performance comparison (D2 vs D3)
- ✅ HiPerGator GPU instructions
- ✅ Quick start guide (3 steps)
- ✅ New features documented

### **Research Poster**
- ✅ A3 or 36×48" PDF
- ✅ Professional layout with all key sections
- ✅ QR code to GitHub
- ✅ Before/after results prominently displayed

### **Model Performance**
- ✅ BLEU improved from 23.1 → 27-30+ (realistic goal with GPU + augmentation + optimization)
- ✅ Validation loss improved
- ✅ Geometric metrics computed on 50+ test cases
- ✅ Validation layer catching 80%+ of issues

### **Working Demo**
- ✅ Streamlit app with new features working
- ✅ Color-coded rooms, scale legend, parameter controls
- ✅ Geometric validation display
- ✅ Example gallery

---

## **TIMELINE & TASK ASSIGNMENT**

| Phase | Days | Task | Owner | Status |
|-------|------|------|-------|--------|
| **Phase 1** | 1-2 | Geometric validation + evaluation notebook | You | ✅ In Progress |
| **Phase 1** | 2 | Create 20 side-by-side examples | You | Not Started |
| **Phase 2** | 2 | Data augmentation implementation | You | Not Started |
| **Phase 2** | 2-3 | GPU hyperparameter tuning | You (HiPerGator) | Not Started |
| **Phase 2** | 4 | Final training run with all optimizations | You (HiPerGator) | Not Started |
| **Phase 3** | 3-4 | UI improvements (color, controls, validation display) | You | Not Started |
| **Phase 3** | 4 | Take UI screenshots for report | You | Not Started |
| **Phase 3** | 4-5 | Write expanded IEEE report (5-6 pages) | You | Not Started |
| **Phase 3** | 4 | Update README with visuals | You | Not Started |
| **Phase 4** | 5 | Design and export research poster | You | Not Started |
| **Final** | 5 | Proofread, finalize, push commits | You | Not Started |

---

## **SUCCESS METRICS FOR FULL POINTS**

**Report Quality:** 5-6 pages, visually compelling, addresses all professor feedback  
**Code Quality:** Clean commits showing progression, no dead code  
**Model Improvement:** BLEU 27-30+ (realistic from augmentation + GPU optimization)  
**Evaluation Depth:** 50+ test cases, error analysis, side-by-side examples  
**UI Polish:** Color-coded, parameter controls, validation display, professional appearance  
**Documentation:** Clear README, comprehensive report, research poster  
**Reproducibility:** GPU setup instructions, hyperparameters, random seeds documented  
**GitHub Evidence:** 5+ commits with clear diffs  

---

## **SUBMISSION CHECKLIST**

- [ ] Geometric validation module fully integrated
- [ ] Comprehensive evaluation notebook created and run
- [ ] Data augmentation pipeline implemented
- [ ] Hyperparameter tuning completed on GPU
- [ ] Final model trained with all optimizations
- [ ] Model performance improved (target: BLEU 27-30+)
- [ ] Streamlit UI updated with new features
- [ ] Screenshots captured for report
- [ ] IEEE report written (5-6 pages)
- [ ] README updated with visual overview and GPU instructions
- [ ] Research poster designed and exported
- [ ] All commits pushed to GitHub
- [ ] Final proofread and quality check
- [ ] Submit GitHub link + report PDF + poster

---

## **PROFESSOR'S KEY FEEDBACK ADDRESSED**

| Feedback Item | Solution | Status |
|---------------|----------|--------|
| Report too short (needs 1-2 more pages) | Write 5-6 page report with visuals | In Progress |
| Need system architecture diagram | Create updated architecture + flowchart | In Progress |
| Need dataset pipeline explanation | Add pipeline diagram showing RPLAN→Code transformation | Planned |
| Need geometric metrics section | Implement GeometricMetrics class + comparison | In Progress |
| Need side-by-side examples | Generate 20+ with commentary | Planned |
| Need UI improvements | Add color-coding, scale legend, controls | Planned |
| Need README visual overview | Add architecture block + performance table | Planned |
| Need geometric validation layer | ✅ **DONE** - `geometric_validator.py` created | Complete |
| Need enhanced evaluation | ✅ **DONE** - `evaluation_enhanced.ipynb` created | Complete |
| Need BLEU >40 attempt | Training on GPU with augmentation + optimization | In Progress |
| Need deployment readiness | Add Docker template, API spec in production guide | Planned |

---

**END OF PLAN**

*Last Updated: November 18, 2025*  
*Status: READY FOR EXECUTION*
