"""
Supabase Cloud Rock Library Utilities
FastAPI version — no Streamlit dependency
Reads credentials from os.environ (loaded from .env by web/main.py)
"""

import os
import io
import numpy as np

# ═══════════════════════════════════════════════════════
# SUPABASE CONNECTION
# ═══════════════════════════════════════════════════════

_supabase_client = None


def _get_client():
    """Lazy-init Supabase client; returns None if not configured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception:
        return None


def is_supabase_available():
    """Return True if Supabase credentials are present and client init works."""
    return _get_client() is not None


# ═══════════════════════════════════════════════════════
# SAMPLE CATALOG
# ═══════════════════════════════════════════════════════

SAMPLE_CATALOG = {
    "MEC Carbonate": {
        "folder": "MEC_Carbonate",
        "regime": "physics",
        "winner": "Kozeny-Carman",
        "samples": {
            "Sample A": {"file": "mec_sample_a.npy", "porosity": 0.13, "desc": "Low porosity carbonate"},
            "Sample B": {"file": "mec_sample_b.npy", "porosity": 0.18, "desc": "Medium porosity section"},
            "Sample C": {"file": "mec_sample_c.npy", "porosity": 0.20, "desc": "High porosity carbonate"},
        },
    },
    "Indiana Limestone (ILS)": {
        "folder": "ILS_Limestone",
        "regime": "physics",
        "winner": "Kozeny-Carman",
        "samples": {
            "Sample A": {"file": "ils_sample_a.npy", "porosity": 0.15, "desc": "Well-connected grainstone"},
            "Sample B": {"file": "ils_sample_b.npy", "porosity": 0.19, "desc": "Medium porosity section"},
            "Sample C": {"file": "ils_sample_c.npy", "porosity": 0.23, "desc": "High porosity grainstone"},
        },
    },
    "Synthetic Blobs": {
        "folder": "Synthetic_Blobs",
        "regime": "physics",
        "winner": "Kozeny-Carman",
        "samples": {
            "Sample A": {"file": "syn_sample_a.npy", "porosity": 0.20, "desc": "Low heterogeneity"},
            "Sample B": {"file": "syn_sample_b.npy", "porosity": 0.25, "desc": "Medium heterogeneity"},
            "Sample C": {"file": "syn_sample_c.npy", "porosity": 0.30, "desc": "High heterogeneity"},
        },
    },
    "Estaillades (Vuggy)": {
        "folder": "Estaillades_Carbonate",
        "regime": "ai",
        "winner": "GNN (+28.4%)",
        "samples": {
            "Sample A": {"file": "est_sample_a.npy", "porosity": 0.12, "desc": "Complex vug network"},
            "Sample B": {"file": "est_sample_b.npy", "porosity": 0.09, "desc": "Dense vuggy section"},
            "Sample C": {"file": "est_sample_c.npy", "porosity": 0.16, "desc": "Mixed vug + matrix"},
        },
    },
    "Savonnières (3-Phase)": {
        "folder": "Savonnieres_Carbonate",
        "regime": "ai",
        "winner": "GNN (+46.2%)",
        "samples": {
            "Sample A": {"file": "sav_sample_a.npy", "porosity": 0.21, "desc": "Multi-scale vugs"},
            "Sample B": {"file": "sav_sample_b.npy", "porosity": 0.10, "desc": "Tight 3-phase section"},
            "Sample C": {"file": "sav_sample_c.npy", "porosity": 0.16, "desc": "Open vuggy region"},
        },
    },
}


# ═══════════════════════════════════════════════════════
# API FUNCTIONS (used by web/routers/rocks.py)
# ═══════════════════════════════════════════════════════

def list_samples():
    """
    Return a flat list of sample dicts for the /api/rocks endpoint.
    Each dict has: id, name, rock_type, porosity, shape, gnn_wins
    """
    results = []
    for rock_name, catalog in SAMPLE_CATALOG.items():
        gnn_wins = catalog["regime"] == "ai"
        for sample_name, info in catalog["samples"].items():
            sample_id = f"{catalog['folder']}/{info['file']}"
            results.append({
                "id": sample_id,
                "name": f"{rock_name} — {sample_name}",
                "rock_type": rock_name,
                "porosity": info["porosity"],
                "shape": [128, 128, 128],
                "gnn_wins": gnn_wins,
                "folder": catalog["folder"],
                "file": info["file"],
                "desc": info["desc"],
            })
    return results


def get_sample_bytes(folder: str, filename: str) -> bytes:
    """
    Download a .npy file from Supabase Storage and return raw bytes.
    Raises RuntimeError if Supabase is unavailable or download fails.
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("Supabase is not configured")

    file_path = f"{folder}/{filename}"
    try:
        response = client.storage.from_("rock-samples").download(file_path)
        return bytes(response)
    except Exception as e:
        raise RuntimeError(f"Failed to download '{file_path}': {e}")


def get_public_url(folder: str, filename: str) -> str | None:
    """Return the public URL for a sample (if bucket is public)."""
    client = _get_client()
    if client is None:
        return None
    file_path = f"{folder}/{filename}"
    try:
        return client.storage.from_("rock-samples").get_public_url(file_path)
    except Exception:
        return None
