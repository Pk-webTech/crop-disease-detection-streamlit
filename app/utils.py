"""
Shared paths and configuration for the Crop Disease AI app.

app/farm_app.py imports MODEL_PATH, CSV_PATH, LOG_PATH, OUTPUT_IMAGE_DIR,
OUTPUT_VIDEO_DIR, OUTPUT_REPORT_DIR, SAMPLE_IMAGE_DIR and SAMPLE_VIDEO_DIR
from this module, so everything referenced there is defined here.
"""

from pathlib import Path

# Project root = parent of the "app" folder this file lives in
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------
# Model / data files
# ----------------------------------------------------------------
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")
CSV_PATH = str(BASE_DIR / "data" / "fertilizer_database.csv")
LOG_PATH = str(BASE_DIR / "data" / "disease_progression_log.csv")

# ----------------------------------------------------------------
# Output directories (created automatically if missing)
# ----------------------------------------------------------------
OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "videos"
OUTPUT_REPORT_DIR = BASE_DIR / "outputs" / "reports"

# ----------------------------------------------------------------
# Sample media directories (for demo images/videos, if you add any)
# ----------------------------------------------------------------
SAMPLE_IMAGE_DIR = BASE_DIR / "samples" / "images"
SAMPLE_VIDEO_DIR = BASE_DIR / "samples" / "videos"

for _dir in (
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    OUTPUT_REPORT_DIR,
    SAMPLE_IMAGE_DIR,
    SAMPLE_VIDEO_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
