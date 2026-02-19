"""
Dashboard Router — GET /api/dashboard-data
Returns 5-dataset performance results as JSON for Plotly.js charts.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/dashboard-data")
async def get_dashboard_data():
    """Return all 5-dataset benchmark results for the dashboard charts."""
    data = {
        "rocks": ["Savonnières", "Estaillades", "MEC Carbonate", "ILS Limestone", "Synthetic"],
        "samples": [191, 176, 398, 266, 200],
        "cv_values": [2.5, 2.8, 0.85, 0.52, 0.45],
        "baseline_mse": [5.545, 0.112, 0.002, 0.025, 0.234],
        "gnn_mse": [2.985, 0.080, 0.337, 0.327, 1.267],
        "improvement": [46.2, 28.4, 0, 0, 0],
        "gnn_wins": [True, True, False, False, False],
        "colors": {
            "primary": "#00FF9D",
            "secondary": "#2D5BFF",
            "muted": "#484f58",
            "bg": "#0E1117",
            "card": "#161B22",
            "border": "#30363D",
            "text": "#E0E0E0",
        },
        "threshold_cv": 1.5,
        "total_samples": 1231,
    }
    return JSONResponse(content=data)
