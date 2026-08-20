"""
Standalone offline script: run YOLOv8 detection on a video and write a
text report of detected diseases + fertilizer recommendations.

Update SOURCE_VIDEO below to point at the video you want to process.
"""

from pathlib import Path
import sys

from ultralytics import YOLO
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
CSV_PATH = BASE_DIR / "data" / "fertilizer_database.csv"
REPORT_PATH = BASE_DIR / "outputs" / "reports" / "result_summary.txt"

# Change this to the video you want to analyze
SOURCE_VIDEO = BASE_DIR / "samples" / "videos" / "sample.mp4"

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Redirect output to a text report
sys.stdout = open(REPORT_PATH, "w", encoding="utf-8")

# 1) Load the trained model
model = YOLO(str(MODEL_PATH))

# 2) Load fertilizer database
fertilizer_db = pd.read_csv(CSV_PATH)

# 3) Build recommendation lookup dictionary
recommendations = {
    row["disease"]: {
        "fertilizer_name": row["fertilizer_name"],
        "type": row["type"],
        "dosage": row["dosage"],
        "brand": row["brand"]
    }
    for _, row in fertilizer_db.iterrows()
}

# 4) Run detection
results = model.predict(
    source=str(SOURCE_VIDEO),
    conf=0.5,
    show=True
)

# 5) Display and save detected results + recommendations
print("🌾 PEST AND DISEASE DETECTION REPORT 🌾\n")
print("========================================\n")

for r in results:
    for c in r.boxes.cls:
        label = model.names[int(c)]
        if label in recommendations:
            rec = recommendations[label]
            print(f"🩺 Detected: {label}")
            print(f"💊 Fertilizer: {rec['fertilizer_name']} ({rec['type']})")
            print(f"📦 Brand: {rec['brand']}")
            print(f"💧 Dosage: {rec['dosage']}\n")
        else:
            print(f"⚠️ No recommendation found for: {label}\n")

print("========================================")
print(f"✅ Report saved successfully at: {REPORT_PATH}")

# Close the file output stream
sys.stdout.close()
