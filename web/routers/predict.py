"""
Predict Router — POST /api/predict + GET /api/predict/progress/{job_id}

Flow:
  1. Client POSTs multipart .npy + rock_type + use_hybrid
  2. Server creates a job_id, starts inference in a background thread
  3. Server returns { job_id } immediately
  4. Client opens EventSource('/api/predict/progress/{job_id}') for SSE updates
  5. Background thread emits progress events + final result via shared queue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uuid
import math
import tempfile
import threading
import asyncio
import queue
import numpy as np
import logging
import traceback

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger (so it uses main.py's config)
logger.propagate = True
logger.setLevel(logging.INFO)

router = APIRouter()

# ── Global State ─────────────────────────────────────────────────────────────
# job_id → {"queue": Queue, "result": dict | None, "error": str | None}
_jobs: dict = {}
_inference_lock = threading.Lock()   # Prevent concurrent GPU/CPU inference


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_k(log_k: float) -> dict:
    if log_k < -15:
        return {"tier": "Ultra-Tight", "color": "#ef4444", "icon": "🔴",
                "quality": "Essentially impermeable",
                "flow": "Almost no fluid can pass through this rock.",
                "analogy": "Like pushing water through a brick wall.",
                "reservoir": "Not viable — would require hydraulic fracturing."}
    elif log_k < -14:
        return {"tier": "Tight",       "color": "#f97316", "icon": "🟠",
                "quality": "Very low permeability",
                "flow": "Extremely restricted; only under high pressure.",
                "analogy": "Similar to shale or very dense limestone.",
                "reservoir": "Challenging — may need stimulation."}
    elif log_k < -13:
        return {"tier": "Low",         "color": "#eab308", "icon": "🟡",
                "quality": "Low to moderate permeability",
                "flow": "Flows slowly; suitable with applied pressure.",
                "analogy": "Comparable to fine sandstone or chalk.",
                "reservoir": "Marginal — viability depends on thickness."}
    elif log_k < -12:
        return {"tier": "Moderate",    "color": "#2D5BFF", "icon": "🔵",
                "quality": "Good permeability",
                "flow": "Fluid flows readily through the pore network.",
                "analogy": "Similar to clean sandstone — a typical reservoir.",
                "reservoir": "Good — suitable for conventional extraction."}
    elif log_k < -11:
        return {"tier": "High",        "color": "#00FF9D", "icon": "🟢",
                "quality": "High permeability",
                "flow": "Flows easily with minimal resistance.",
                "analogy": "Like coarse-grained sand or connected vugs.",
                "reservoir": "Excellent — high flow rates expected."}
    else:
        return {"tier": "Very High",   "color": "#00FF9D", "icon": "🟢",
                "quality": "Very high permeability",
                "flow": "Essentially unrestricted fluid flow.",
                "analogy": "Like gravel or fractured rock.",
                "reservoir": "Exceptional — very high production potential."}


def _run_inference(job_id: str, tmp_path: str, rock_type: str, use_hybrid: bool, porosity: float):
    """Runs in a background thread. Emits progress via queue."""
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def emit(step: str, pct: int, detail: str = ""):
        q.put({"step": step, "pct": pct, "detail": detail})

    try:
        # Force output to stdout immediately (for visibility)
        print(f"\n{'='*60}")
        print(f"INFERENCE JOB STARTED: {job_id}")
        print(f"Rock: {rock_type}, Hybrid: {use_hybrid}, Porosity: {porosity:.3f}")
        print(f"{'='*60}\n")
        
        logger.info("="*60)
        logger.info(f"Starting inference job {job_id}")
        logger.info(f"  Rock: {rock_type}, Hybrid: {use_hybrid}, Porosity: {porosity:.3f}")
        logger.info("="*60)
        emit("Loading model weights", 10, f"Rock: {rock_type}")
        emit("Analyzing voxel structure", 20, "128^3 voxels")
        emit("Extracting pore network (SNOW2)", 45, "Finding pore centers...")
        emit("Building graph topology", 60, "Nodes = pores, Edges = throats")
        emit("Running GNN inference", 80, "GraphSAGE forward pass")
        emit("Computing permeability", 90, "Converting to m^2")

        with _inference_lock:
            logger.info(f"Calling predict_single_chunk for job {job_id}")
            from src.inference import predict_single_chunk
            predicted_k, image_path, baseline_k = predict_single_chunk(
                tmp_path,
                output_image=f"web/static/img/result_{job_id}.png",
                use_hybrid=use_hybrid,
                rock_type=rock_type,
            )
            logger.info(f"Inference completed for job {job_id} - K: {predicted_k:.4e} m^2")

        log_k = math.log10(predicted_k) if predicted_k > 0 else -20
        k_darcy = predicted_k / 9.869233e-13
        k_mdarcy = k_darcy * 1000
        scale_pct = max(0, min(100, ((log_k + 18) / 8) * 100))
        classification = _classify_k(log_k)

        porosity_label = ("high" if porosity > 0.25
                          else "moderate" if porosity > 0.15 else "low")
        porosity_note = (
            f"High porosity with {classification['tier'].lower()} permeability "
            "suggests poor pore connectivity."
            if porosity > 0.2 and log_k < -14
            else "Porosity and permeability are consistent for this rock type."
        )

        result = {
            "permeability": predicted_k,
            "baseline_k": baseline_k,
            "log_k": log_k,
            "k_mdarcy": k_mdarcy,
            "scale_pct": scale_pct,
            "porosity": porosity,
            "porosity_label": porosity_label,
            "porosity_note": porosity_note,
            "rock_type": rock_type,
            "use_hybrid": use_hybrid,
            "image_path": f"/static/img/result_{job_id}.png" if image_path and os.path.exists(image_path) else None,
            **classification,
        }

        if use_hybrid and baseline_k and baseline_k > 0:
            result["correction_pct"] = ((baseline_k - predicted_k) / baseline_k) * 100
            result["baseline_log_k"] = math.log10(baseline_k)

        job["result"] = result
        emit("Complete", 100, "Prediction finished")
        q.put({"done": True, "result": result})
        
        # Force success output to stdout
        print(f"\n{'='*60}")
        print(f"INFERENCE COMPLETE: {job_id}")
        print(f"Predicted K: {predicted_k:.4e} m^2")
        print(f"{'='*60}\n")
        
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        # Force error output to stdout immediately
        print(f"\n{'='*60}")
        print(f"ERROR in inference job {job_id}:")
        print(f"{error_msg}")
        print(f"{'='*60}")
        print(f"Full traceback:\n{error_traceback}")
        print(f"{'='*60}\n")
        
        logger.error(f"Inference failed for job {job_id}: {error_msg}")
        logger.error(f"Traceback:\n{error_traceback}")
        
        # Truncate error message if too long (for display)
        display_error = error_msg[:500] if len(error_msg) > 500 else error_msg
        job["error"] = display_error
        
        # Send error through queue with proper format (non-blocking)
        try:
            # Try to put error in queue, but don't block if queue is full
            q.put_nowait({
                "error": display_error,
                "step": f"Error: {display_error[:80]}...",
                "pct": 0,
                "detail": "Check server logs for full traceback"
            })
        except queue.Full:
            logger.warning(f"Queue full, could not send error to queue for job {job_id}")
        except Exception as queue_err:
            logger.warning(f"Error sending to queue: {queue_err}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    rock_type: str = Form("MEC"),
    use_hybrid: str = Form("false"),
):
    """Accept .npy upload, start inference in background, return job_id."""
    try:
        print(f"\n[PREDICT] Received request: {file.filename}, Rock: {rock_type}, Hybrid: {use_hybrid}")
        logger.info(f"Received prediction request - File: {file.filename}, Rock: {rock_type}, Hybrid: {use_hybrid}")
        
        if not file.filename.endswith(".npy"):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=422, detail="Only .npy files are accepted.")

        contents = await file.read()
        logger.debug(f"File size: {len(contents)} bytes")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        logger.debug(f"Saved to temp file: {tmp_path}")

        # Quick porosity estimate from the array
        try:
            arr = np.load(tmp_path)
            porosity = float(np.mean(arr > 0))
            logger.info(f"Chunk shape: {arr.shape}, Porosity: {porosity:.3f}")
        except Exception as e:
            logger.warning(f"Failed to estimate porosity: {e}")
            porosity = 0.0

        job_id = str(uuid.uuid4())
        _jobs[job_id] = {"queue": queue.Queue(), "result": None, "error": None}
        print(f"[PREDICT] Created job: {job_id}")
        logger.info(f"Created job {job_id}")

        # Fire and forget in background thread
        t = threading.Thread(
            target=_run_inference,
            args=(job_id, tmp_path, rock_type, use_hybrid.lower() == "true", porosity),
            daemon=True,
        )
        t.start()
        print(f"[PREDICT] Started inference thread for job {job_id}")
        logger.info(f"Started inference thread for job {job_id}")

        return JSONResponse(content={"job_id": job_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/predict/progress/{job_id}")
async def predict_progress(job_id: str):
    """SSE endpoint: streams progress events until inference is done."""
    logger.info(f"SSE connection opened for job {job_id}")
    
    if job_id not in _jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    async def event_generator():
        import json
        try:
            while True:
                try:
                    msg = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: q.get(timeout=60)
                    )
                    if "error" in msg:
                        error_text = msg.get('error', 'Unknown error')
                        logger.error(f"Job {job_id} error: {error_text}")
                        yield f"event: error\ndata: {json.dumps({'error': error_text, 'step': msg.get('step', 'Error'), 'pct': msg.get('pct', 0)})}\n\n"
                        break
                    elif msg.get("done"):
                        logger.info(f"Job {job_id} completed")
                        yield f"event: complete\ndata: {json.dumps(msg)}\n\n"
                        break
                    else:
                        logger.debug(f"Job {job_id} progress: {msg.get('step')} ({msg.get('pct')}%)")
                        yield f"event: progress\ndata: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    # Check if job has error set (might have happened before SSE connected)
                    if job.get("error"):
                        error_text = job["error"]
                        logger.error(f"Job {job_id} has error (detected on timeout): {error_text}")
                        yield f"event: error\ndata: {json.dumps({'error': error_text, 'step': 'Error', 'pct': 0})}\n\n"
                        break
                    logger.warning(f"SSE timeout for job {job_id} - no messages received")
                    yield f"event: timeout\ndata: {json.dumps({'error': 'Request timeout - inference may still be running'})}\n\n"
                    break
                except Exception as e:
                    logger.error(f"SSE generator exception for job {job_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Check job error state
                    if job.get("error"):
                        yield f"event: error\ndata: {json.dumps({'error': job['error'], 'step': 'Error', 'pct': 0})}\n\n"
                    else:
                        yield f"event: error\ndata: {json.dumps({'error': f'SSE error: {str(e)}'})}\n\n"
                    break
        except Exception as e:
            logger.error(f"SSE generator error for job {job_id}: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Cleanup job after delivery
            if job_id in _jobs:
                logger.debug(f"Cleaning up job {job_id}")
                del _jobs[job_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/predict/status/{job_id}")
async def predict_status(job_id: str):
    """Get current status of a prediction job (for error recovery)."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    return JSONResponse(content={
        "job_id": job_id,
        "result": job.get("result"),
        "error": job.get("error"),
        "status": "completed" if job.get("result") else ("error" if job.get("error") else "running")
    })
