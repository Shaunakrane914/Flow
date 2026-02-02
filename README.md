# 🪨 Topo-Flow: Graph Neural Networks for Permeability Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)

**When does rock topology matter for permeability prediction?** This project answers that question with Graph Neural Networks trained on **5 diverse rock types** (1,231 samples).

## 🎯 Key Finding

**GNN wins on vuggy carbonates (Cv > 1.5), Kozeny-Carman wins on uniform rocks (Cv < 1.5)**

| Rock Type | Samples | Result | Improvement |
|-----------|---------|--------|-------------|
| 🔥 **Savonnières** | 191 | **GNN Wins** | **+46.2%** |
| 🏆 **Estaillades** | 176 | **GNN Wins** | **+28.4%** |
| 📊 MEC Carbonate | 398 | Baseline Wins | -17.3% |
| 📏 ILS Limestone | 266 | Baseline Wins | -13.0% |
| 🧪 Synthetic | 200 | Baseline Wins | -34.1% |

**Pattern:** Topology matters when heterogeneity (Cv) exceeds **1.5**

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Shaunakrane914/Flow.git
cd Flow

# Install dependencies
pip install -r requirements.txt
```

### Run Web App

```bash
streamlit run app.py
```

Upload a 128³ rock chunk (.npy file) and get instant permeability predictions!

---

## 📊 What's Inside

### **Core Components**

- **`app.py`** - Streamlit web interface (5 rock types, hybrid mode)
- **`src/model.py`** - Standard GNN architecture (GraphSAGE)
- **`src/model_hybrid.py`** - Hybrid physics-informed model
- **`src/physics.py`** - Nuclear pore extraction + Stokes flow simulation
- **`src/graph_extraction.py`** - SNOW algorithm + PyG graph conversion

### **Trained Models** (`models/`)

- `best_model_savonnieres.pth` - Savonnières vuggy carbonate (**best: +46%**)
- `best_model_estaillades.pth` - Estaillades vuggy carbonate (+28%)
- `best_model_hybrid.pth` - Hybrid model (MEC)
- `best_model_ils.pth` - Indiana Limestone
- `best_model_synthetic.pth` - Synthetic rocks
- `best_model.pth` - MEC carbonate

### **Training Scripts** (`src/`)

```bash
# Train on each dataset
python src/train_savonnieres.py
python src/train_estaillades.py
python src/train_ils.py
python src/train_synthetic.py
python src/train_hybrid.py  # MEC hybrid model

# Compare against baselines
python src/baseline_savonnieres.py
python src/baseline_estaillades.py
python src/baseline_ils.py
python src/baseline_synthetic.py
```

---

## 🔬 Technical Approach

### **1. Nuclear Pore Extraction**

Novel algorithm achieving **100% success rate** on vuggy rocks (vs 47% for traditional methods):

```python
from src.physics import get_permeability

# Extract permeability using Stokes flow
permeability = get_permeability(pore_network, chunk_shape)
```

### **2. Graph Neural Network**

**Architecture:**
- GraphSAGE (3 layers: 128→64→32→16)
- Global mean pooling
- Node features: log(diameter), log(volume)
- Edge features: throat connections

**Key Innovation:** Learns **which vugs connect** to flow network

### **3. Hybrid Model**

For MEC dataset only:
```
K_hybrid = K_baseline + Δ_GNN
```

Where:
- K_baseline = Calibrated Kozeny-Carman
- Δ_GNN = Residual correction from GNN

**Result:** Never worse than baseline (+0.1% improvement)

---

## 📈 Results

### **The Dual-Regime Framework**

```
       GNN Improvement (%)
    +50% |        🔥 Savonnières (+46%)
         |
    +30% |    🏆 Estaillades (+28%)
         |
     0%  |────────────── Cv = 1.5 ──────────
         |          📊 MEC (-17%)
   -20%  |       📏 ILS (-13%)
         |
   -40%  |    🧪 Synthetic (-34%)
         └──────────────────────────────────→
           0.5    1.0    1.5   2.0   2.5   Heterogeneity (Cv)
```

**Critical Threshold:** Cv = 1.5 (coefficient of variation)

### **When to Use GNN vs Kozeny-Carman?**

```python
if heterogeneity_index > 1.5:
    # Vuggy, complex pore network
    use_GNN()  # 20-50% better
else:
    # Uniform, well-connected pores
    use_Kozeny_Carman()  # Simpler, equally accurate
```

---

## 🎓 Scientific Contribution

### **Not "AI hype" - We show when ML fails too!**

Unlike typical ML papers claiming "AI beats everything," we demonstrate:

1. ✅ **GNN wins on 2/5 datasets** (both vuggy carbonates)
2. ✅ **Baseline wins on 3/5 datasets** (uniform rocks)
3. ✅ **Reproducible threshold** (Cv = 1.5, standard metric)
4. ✅ **Practical decision framework** for engineers

### **Publication Strategy**

**Target:** Water Resources Research (IF: 5.4)

**Title:** *"Graph Neural Networks for Permeability Prediction: When Does Topology Matter?"*

**Key Message:**
> "We don't claim GNN always wins. We identify the complexity threshold where topology-aware ML beats classical formulas."

---

## 📁 Project Structure

```
Flow/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Dependencies
├── models/                     # Trained GNN weights (.pth)
│   ├── best_model_savonnieres.pth
│   ├── best_model_estaillades.pth
│   └── ...
├── src/                        # Core algorithms
│   ├── model.py               # GNN architecture
│   ├── model_hybrid.py        # Hybrid model
│   ├── physics.py             # Nuclear extraction + Stokes
│   ├── graph_extraction.py    # SNOW + PyG conversion
│   ├── train_*.py             # Training scripts (5)
│   ├── baseline_*.py          # Comparison scripts (5)
│   ├── process_*.py           # Data processing (4)
│   └── inference.py           # Prediction pipeline
└── data/                       # Datasets (NOT in repo)
    ├── graphs_savonnieres/    # 191 graphs
    ├── graphs_estaillades/    # 176 graphs
    ├── graphs_ils/            # 266 graphs
    ├── graphs_nuclear/        # 398 MEC graphs
    └── graphs_synthetic/      # 200 graphs
```

**Note:** `data/` folder is excluded via `.gitignore` (large datasets)

---

## 🛠️ Requirements

```
torch==2.7.1
torch-geometric==2.3.1
porespy==3.0.3
openpnm==3.6.0
streamlit==1.30.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

**Python:** 3.8+  
**GPU:** Optional (CUDA 11.8+ for faster training)

---

## 📊 Example Usage

### **Command Line Prediction**

```python
from src.inference import predict_single_chunk

# Predict permeability for a rock chunk
k_predicted, viz_path, k_baseline = predict_single_chunk(
    chunk_path="chunk.npy",
    rock_type="Savonnieres",
    use_hybrid=False
)

print(f"Predicted K: {k_predicted:.2e} m²")
```

### **Web Interface**

```bash
streamlit run app.py
```

1. Select rock type (MEC, ILS, Synthetic, Estaillades, Savonnières)
2. Upload 128³ .npy chunk (binary: 0=solid, 1=pore)
3. Get instant prediction + 3D visualization

---

## 🏆 Highlights

- ✅ **5 rock types**, 1,231 total samples
- ✅ **100% pore extraction success** (Nuclear algorithm)
- ✅ **46% improvement** on Savonnières (best result)
- ✅ **Dual-regime framework** (practical decision tool)
- ✅ **Publication-ready** results
- ✅ **Live web app** (Streamlit)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

**Datasets:**
- MEC Carbonate: High-resolution micro-CT scan
- Indiana Limestone (ILS): Public reservoir database
- Estaillades: French outcrop carbonate
- Savonnières: 3-phase vuggy carbonate

**Tools:**
- PyTorch Geometric for GNN framework
- PoreSpy/OpenPNM for pore network extraction
- Streamlit for web interface

---

## 📧 Contact

**Author:** Shaunak Rane  
**GitHub:** [@Shaunakrane914](https://github.com/Shaunakrane914)

---

**⭐ If this project helps your research, please cite and star the repo!**
