"""Compatibility entry point for validated TopoFlow experiments.

Requires an explicit manifest and fresh output directory. Dataset membership,
parent groups and all three splits are declared in the manifest, not inferred
from the historical script name. See experiments/README.md.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiments import main

if __name__ == '__main__':
    main()
