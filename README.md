# CADlingo: Text-to-CAD Generation System

CADlingo turns natural language floor-plan descriptions into validated AutoCAD commands, DXF exports, and preview images via a fine-tuned CodeT5 model trained on the RPLAN dataset. The production-ready stack pairs Dockerized FastAPI services, a Streamlit UI, and geometric validation (IoU, adjacency, overlap checks) so teams can prototype plans without manual CAD scripting.

## Refined Pipeline (local)
1. ```bash
   python -m venv cad-env
   source cad-env/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```
2. Configure Kaggle and place the RPLAN pickle files under `data/raw/pickle/` (see `docs/SETUP_GUIDE.md`).
3. ```bash
   python src/data/dataset_creator.py
   ```
4. ```bash
   python src/models/train.py
   ```
   Adjust parameters (batch size, epochs, LR) via the script flags.
5. ```bash
   python src/models/inference.py --description "A 1200 sq ft plan with 2 bedrooms"
   ```
6. Optional: ```bash
   ./run.sh
   ```
   Use the menu to repeat dataset creation, training, inference, and UI launches.

## Launching the Interface
- Streamlit UI: ```bash
  streamlit run ui/app.py
  ``` (visit `http://localhost:8501` to enter prompts, adjust beams/temperature, and download DXF/TXT output).
- FastAPI service: ```bash
  uvicorn ui.api:app --reload
  ``` (endpoints `/generate`, `/generate-with-validation`, `/generate-dxf`, `/batch-generate`, `/health`, `/metrics`; docs at `http://localhost:8000/docs`).
- Production Docker: ```bash
  docker-compose up -d
  ``` (starts Streamlit, FastAPI, and helper services; monitor via `docker-compose logs`).

## Project Structure
```
CADlingo/
├── data/
│   ├── raw/                  # RPLAN pickles & downloads
│   ├── processed/            # generated train/val datasets
│   └── outputs/              # inference code, DXF, PNG
├── src/
│   ├── data/                 # dataset_creator, downloader, rplan_loader
│   └── models/               # train.py, inference.py, validators
├── ui/
│   ├── app.py                # Streamlit 
│   └── api.py                # FastAPI service entrypoints
├── docs/                     # SETUP_GUIDE, ARCHITECTURE, deployment notes
├── notebooks/                # setup.ipynb, train_and_evaluate.ipynb
├── results/                  # models/, plots/, samples/
├── run.sh                    # helper for dataset/training/UI tasks
├── requirements.txt          # pinned Python dependencies
└── Dockerfile & docker-compose.yml
```

## Performance Snapshot
| Metric | Current | Target |
|--------|---------|--------|
| BLEU score | 23.1 | >40 |
| Validation loss | 2.1 | <1.0 |
| IoU (layout overlap) | 0.85 | >0.90 |
| Room count accuracy | 0.94 | >0.95 |
| Adjacency accuracy | 0.81 | >0.85 |
| Overall quality score | 88.3% | >90% |
| Inference latency | 1–2 s | <3 s |

## Known Issues & Warnings
- BLEU remains low when training on ~1,000 samples; increase `train_samples` in `src/data/dataset_creator.py` and retrain for better fluency.
- Geometry is limited to axis-aligned rectangles; refine complex shapes manually in a CAD editor.
- Complex prompts can still trigger overlap/adjacency warnings—review the validation report and PNG preview before exporting.
- Training uses ~8 GB RAM and benefits from a GPU; drop the batch size to 4 on constrained systems.
- Streamlit/FastAPI take 10–30 seconds to cache checkpoints on the first request.

## Author & Contact
- Ramya Lakshmi Kuppa Sundararajan (University of Florida, Applied Data Science)
- Email: `ramyalakshmi.ks@gmail.com`
- GitHub: `https://github.com/RamyaLakshmiKS/CADlingo`
- LinkedIn: `https://www.linkedin.com/in/ramyalakshmiks`
- Issues & feature requests: `https://github.com/RamyaLakshmiKS/CADlingo/issues`

Include your OS, Python version, and whether you are using Docker or a local virtual environment when requesting help.

## License
This project is available under the [MIT License](LICENSE).

## Acknowledgements
- RPLAN pickle dataset contributors and Kaggle community
- CodeT5 / Hugging Face Transformers team
- PyTorch and ezdxf library maintainers for model and DXF tooling support