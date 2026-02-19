# TopoFlow GNN

> Graph Neural Network for Rock Permeability Prediction  
> Topological Threshold Discovery · 1,231 samples · 5 geological formations

---

## The Discovery

A **Topological Threshold** at Cv ≈ 1.5 (pore size coefficient of variation) determines whether GNN or the Kozeny-Carman physics formula gives better permeability predictions:

| Rock Type | Cv | Winner | Improvement |
|---|---|---|---|
| Savonnières | ~2.5 | 🤖 GNN | +46.2% |
| Estaillades | 2.80 | 🤖 GNN | +28.4% |
| MEC Carbonate | 0.85 | 📐 Physics | — |
| ILS Limestone | 0.52 | 📐 Physics | — |
| Synthetic | 0.45 | 📐 Physics | — |

---

## Project Structure

```
├── web/                    # FastAPI application (backend + frontend)
│   ├── main.py             # FastAPI entry point
│   ├── routers/            # API route handlers
│   │   ├── predict.py      # POST /api/predict (SSE streaming inference)
│   │   ├── visualize.py    # POST /api/visualize (server-side downsample)
│   │   ├── rocks.py        # GET  /api/rocks (Supabase cloud library)
│   │   └── dashboard.py    # GET  /api/dashboard-data (benchmark JSON)
│   ├── static/
│   │   ├── css/style.css   # Bio-Digital design system
│   │   └── js/             # app.js · predictor.js · viz3d.js · dashboard.js
│   └── templates/          # index.html · home · predictor · dashboard · methodology
│
├── src/                    # Core ML library (used by FastAPI routers)
│   ├── model.py            # GraphSAGE GNN architecture
│   ├── inference.py        # End-to-end inference pipeline
│   ├── graph_extraction.py # SNOW2 → PyG graph conversion
│   ├── physics.py          # Kozeny-Carman baseline
│   ├── visualize.py        # 3D voxel → pore-network helpers
│   ├── supabase_utils.py   # Cloud sample catalog & download
│   └── ...                 # Training, baseline, and analysis scripts
│
├── models/                 # Trained .pth model weights
├── data/                   # Raw micro-CT .npy chunks (gitignored)
├── results/                # Benchmark charts & result summaries
├── scripts/                # One-off data processing utilities
│
├── .env                    # Supabase credentials (gitignored)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation & Quick Start

### 1. Prerequisites

- **Python**: 3.10 (recommended)
- **OS**: Linux, macOS, or Windows 10/11
- **GPU (optional but recommended)**: CUDA-capable GPU for faster GNN inference  
  The project will still run on CPU, just slower for large batches.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Configure Supabase

If you want to use the **Cloud Library** tab (pre-hosted rock samples), create a `.env` file:

```bash
cp .env.example .env
```

Then fill in:

- `SUPABASE_URL`
- `SUPABASE_KEY`

If these are not set, the app will fall back to local/demo behaviour where possible.

### 5. Start the FastAPI web app

```bash
uvicorn web.main:app --port 8502 --reload
```

Then open your browser at:

```text
http://localhost:8502
```

Use the top navbar to switch between:

- `Home` – overview and explanation of the method
- `Predictor` – upload or pick a rock, visualize, and run GNN prediction
- `Dashboard` – benchmark and regime comparison
- `Methodology` – detailed scientific background

---

## Architecture

```
Browser (SPA)
  │── fetch /fragment/{page}  ──► FastAPI ──► Jinja2 HTML template
  │── POST  /api/predict       ──► src/inference.py (GNN)
  │── EventSource /api/predict/progress  (SSE streaming)
  │── POST  /api/visualize     ──► numpy downsample → Plotly.js JSON
  └── GET   /api/rocks         ──► Supabase storage proxy
```

**Tech stack:** FastAPI · PyTorch · PyTorch Geometric · PoreSpy · Plotly.js · Supabase

---

## Results

See [`results/`](results/) for benchmark charts and per-dataset summaries.

Full methodology and interactive dashboard: run the app and visit `#methodology` / `#dashboard`.
