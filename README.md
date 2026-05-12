# LLM Hallucinations

A geometric dynamical systems framework for understanding LLM hallucinations through task-dependent basin structures in latent space.

## Overview

This repository contains experimental code for LLM Hallucination controls. Please feel free to explore the codebase and raise a PR/star or play around with the code!

## Repository Structure

```
.
├── code/                         # Main source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration and hyperparameters
│   ├── data.py                   # Data loading and preprocessing
│   ├── models.py                 # Model loading utilities
│   ├── geometry.py               # Geometric calculations (basins, distances)
│   ├── causality.py              # Causal intervention experiments
│   ├── basins.py                 # Multi-basin clustering and detection
│   ├── experiments.py            # Basin example figure generation
│   ├── visualization.py          # Layer evolution visualization
│   ├── steering.py               # Steering mechanism and results
│   └── requirements.txt          # Python dependencies
│
│
├── data/                          # Data files (download separately)
│   └── [HaluEval, MuSiQue, FEVER, TruthfulQA datasets]
│
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── LICENSE                        # License (add as needed)
```

## Setup

### Requirements
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+
- NumPy, SciPy, scikit-learn
- Matplotlib, seaborn for visualization

### Installation

```bash
# Clone repository
git clone <repo-url>
cd llm-hallucinations

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r code/requirements.txt
```

### Data Setup

Download datasets from their respective sources:
- **HaluEval**: https://github.com/RUCAIBox/HaluEval
- **MuSiQue**: https://github.com/StonyBrookNLP/musique
- **FEVER**: https://fever.ai/dataset/fever.html
- **TruthfulQA**: https://github.com/sylinrl/TruthfulQA

Place them in the `data/` directory following the structure expected by `code/data.py`.

## Core Modules

### `geometry.py`
Computes geometric properties of hidden state representations:
- Basin centroid estimation
- Mahalanobis distance calculations
- Variance ratio analysis for task complexity
- Reference state construction from uninformative contexts

### `causality.py`
Causal intervention experiments:
- Linear interpolation from factual → hallucination basins
- In-model steering vector injection
- Logistic regression classifier training
- Dose-response curve analysis

### `basins.py`
Multi-basin detection for misconception tasks:
- Gaussian mixture modeling of hallucinated states
- Voronoi tessellation partitioning
- Basin assignment probabilities
- K-means clustering for basin discovery

### `steering.py`
Adaptive steering mechanism:
- Geometry-aware steering vector computation
- Risk-signature feature extraction (distance + contraction ratio)
- Dynamic lambda scaling based on basin proximity
- Comprehensive results visualization

### `visualization.py`
Layer-wise evolution visualization:
- 2D and 3D PCA projections
- Hidden state trajectory plotting
- Factual vs. hallucinated state separation
- Multi-layer analysis across model depth

### `experiments.py`
Basin example figure generation:
- Task-dependent basin geometry across datasets
- Multi-model comparison
- Performance metrics per task type

## Running Experiments

### Generate Figures

```bash
cd code

# Generate basin examples 
python -c "from experiments import *; generate_basin_examples()"

# Generate causality intervention figures
python -c "from causality import *; generate_causality_figures()"

# Generate layer evolution visualizations
python -c "from visualization import *; generate_layer_evolution()"

# Generate steering results
python -c "from steering import *; generate_steering_comprehensive()"

# Generate multi-basin Voronoi-paritioning plots
python -c "from basins import *; generate_multi_basin_voronoi()"
```

### Full Experimental Pipeline

The legacy full pipeline script has been archived to `old/core.py` and is not part of the active `code/` module set.

## Reproducibility

All experiments use fixed random seeds (seed=42) for deterministic results:
- NumPy, PyTorch, scikit-learn seeded identically
- 70/30 stratified train/test splits
- All reference statistics computed on training data only
- Three independent random splits with bootstrap confidence intervals

Hyperparameters documented in `code/config.py`

## Contact

Questions? Contact Me!
(kcherukuri@imsa.edu)
(kalyan.cherukuri5@gmail.com)
