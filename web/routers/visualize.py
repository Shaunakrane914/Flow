"""
Visualize Router — POST /api/visualize
Accepts .npy bytes, server-downsamples, returns lightweight JSON for Plotly.js.
This avoids browser memory blowout on large tensors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logger.propagate = True  # Use root logger config from main.py
logger.setLevel(logging.INFO)
router = APIRouter()

# Quality steps (same as viz.py)
QUALITY_STEPS = {"low": 10, "balanced": 6, "hd": 4}


@router.post("/visualize")
async def visualize(
    file: UploadFile = File(...),
    quality: str = Form("balanced"),
):
    """
    Downsample a .npy voxel array server-side and return sparse JSON
    safe for browser Plotly.js consumption.
    """
    step = QUALITY_STEPS.get(quality.lower(), 6)

    # Write upload to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        arr = np.load(tmp_path)
        logger.info(f"Loaded array shape: {arr.shape}, dtype: {arr.dtype}")
    finally:
        os.unlink(tmp_path)

    # Downsample first
    vol = arr[::step, ::step, ::step].astype(np.float32)
    sx, sy, sz = vol.shape
    logger.info(f"Downsampled shape: ({sx}, {sy}, {sz}), step: {step}")
    
    # Apply Gaussian smoothing for organic, smooth pore shapes (fixes jagged geometry)
    # This makes pores look like continuous tubes/caves instead of sharp blocks
    logger.info("Applying Gaussian smoothing for organic pore geometry...")
    vol = gaussian_filter(vol, sigma=1.0, mode='constant')
    logger.info("Gaussian smoothing applied - pores now have smooth, organic shapes")

    # For binary arrays (0/1), use > 0. For float arrays, use > 0.1
    is_binary = np.array_equal(vol, vol.astype(bool))
    threshold = 0.0 if is_binary else 0.1
    logger.info(f"Array is binary: {is_binary}, using threshold: {threshold}")

    # Plotly volume trace needs a regular grid, but we can send sparse for scatter3d fallback
    # Create coordinate grids
    X, Y, Z = np.mgrid[:sx, :sy, :sz]
    flat_val = vol.flatten()

    # For volume trace: send full grid (but limit size for performance)
    # If grid is too large (> 1M points), use sparse mode
    total_points = sx * sy * sz
    use_sparse = total_points > 1000000  # 1M points threshold
    
    if use_sparse:
        logger.info(f"Grid too large ({total_points} points), using sparse mode")
        # Sparse mode: only non-zero voxels
        mask = flat_val > threshold
        num_pores = int(mask.sum())
        logger.info(f"Found {num_pores} pore voxels (out of {flat_val.size} total)")
        
        if num_pores == 0:
            logger.warning("No pore voxels found after thresholding - array may be all solid")
        
        x_out = X.flatten()[mask].tolist()
        y_out = Y.flatten()[mask].tolist()
        z_out = Z.flatten()[mask].tolist()
        v_out = flat_val[mask].tolist()
        rendered_voxels = num_pores
    else:
        logger.info(f"Using full grid mode ({total_points} points)")
        # Full grid mode: send all points (Plotly volume trace needs this)
        x_out = X.flatten().tolist()
        y_out = Y.flatten().tolist()
        z_out = Z.flatten().tolist()
        v_out = flat_val.tolist()
        rendered_voxels = int((flat_val > threshold).sum())

    phi = float(np.mean(arr > 0))
    total_voxels = int(arr.size)

    return JSONResponse(content={
        "x": x_out,
        "y": y_out,
        "z": z_out,
        "values": v_out,
        "shape": [sx, sy, sz],
        "porosity": phi,
        "total_voxels": total_voxels,
        "rendered_voxels": rendered_voxels,
        "step": step,
        "sparse_mode": use_sparse,  # Flag to tell frontend which rendering mode to use
    })
