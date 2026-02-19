"""
Rocks Router — Cloud Library
GET  /api/rocks            — list samples from Supabase
GET  /api/rocks/{id}       — proxy-download .npy bytes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

router = APIRouter()


@router.get("/rocks")
async def list_rocks():
    """List available cloud rock samples. Falls back to mock data if Supabase is unconfigured."""
    try:
        from src.supabase_utils import list_samples
        return JSONResponse(content=list_samples())
    except Exception:
        return JSONResponse(content=_mock_samples())


@router.get("/rocks/{folder}/{filename}")
async def get_rock(folder: str, filename: str):
    """
    Download a .npy file from Supabase and stream raw bytes to the browser.
    sample_id is encoded as two path segments: /api/rocks/{folder}/{filename}
    """
    try:
        from src.supabase_utils import get_sample_bytes
        data = await asyncio.get_event_loop().run_in_executor(
            None, get_sample_bytes, folder, filename
        )
        return StreamingResponse(
            iter([data]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _mock_samples():
    """Return representative mock samples when Supabase is unavailable."""
    return [
        {"id": "sav_001", "name": "Savonnières #1", "rock_type": "Savonnières",
         "porosity": 0.312, "shape": [128, 128, 128], "gnn_wins": True},
        {"id": "est_001", "name": "Estaillades #1", "rock_type": "Estaillades",
         "porosity": 0.128, "shape": [128, 128, 128], "gnn_wins": True},
        {"id": "mec_001", "name": "MEC Carbonate #1", "rock_type": "MEC",
         "porosity": 0.189, "shape": [128, 128, 128], "gnn_wins": False},
        {"id": "ils_001", "name": "ILS Limestone #1", "rock_type": "ILS",
         "porosity": 0.214, "shape": [128, 128, 128], "gnn_wins": False},
        {"id": "syn_001", "name": "Synthetic #1", "rock_type": "Synthetic",
         "porosity": 0.251, "shape": [128, 128, 128], "gnn_wins": False},
    ]
