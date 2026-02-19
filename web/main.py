"""
TopoFlow GNN — FastAPI Entry Point
Bio-Digital Theme: #0E1117 + #00FF9D + #2D5BFF
"""

import os
import sys
import logging

# Ensure project root is on path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Logging Configuration ──────────────────────────────────────────────────────
# Configure root logger BEFORE importing routers (so they inherit the config)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add console handler (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Add file handler
try:
    file_handler = logging.FileHandler('topoflow.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)
    root_logger.addHandler(file_handler)
except Exception as e:
    print(f"Warning: Could not create log file: {e}")

logger = logging.getLogger(__name__)
logger.info("Logging configured - all logs will appear in terminal and topoflow.log")

# Load .env (replaces .streamlit/secrets.toml)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    logger.info("Loaded .env file")
except ImportError:
    logger.debug("python-dotenv not available, skipping .env load")
except Exception as e:
    logger.warning(f"Failed to load .env: {e}")


from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Import routers AFTER logging is configured
from web.routers import predict, rocks, dashboard, visualize

logger.info("="*60)
logger.info("Starting TopoFlow FastAPI application")
logger.info("="*60)

# ── App Init ──────────────────────────────────────────
app = FastAPI(
    title="TopoFlow GNN",
    description="Graph Neural Network for Rock Permeability Prediction",
    version="4.0.0",
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ── Static Files ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ── Templates ─────────────────────────────────────────
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Include Routers ───────────────────────────────────
app.include_router(predict.router, prefix="/api")
app.include_router(rocks.router,   prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(visualize.router, prefix="/api")


# ── Page Routes ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "home"
    })


@app.get("/fragment/{page}", response_class=HTMLResponse)
async def fragment(request: Request, page: str):
    """Return just the page fragment for SPA routing (no shell)."""
    valid_pages = {"home", "predictor", "dashboard", "methodology"}
    if page not in valid_pages:
        page = "home"
    return templates.TemplateResponse(f"{page}.html", {"request": request})


# ── Dev Runner ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Configure uvicorn logging to use our logger
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
    uvicorn.run("web.main:app", host="0.0.0.0", port=8502, reload=True, log_config=log_config)
