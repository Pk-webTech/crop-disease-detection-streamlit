# 🌾 Crop Disease AI

AI-powered crop disease detection system using YOLOv8.

---

## Features

- Crop disease detection (image & video)
- Fertilizer recommendation
- Severity estimation
- Disease progression prediction
- Offline farmer chatbot

---

## Project Structure

```
crop-disease-ai/
├── app/
│   ├── farm_app.py       # Streamlit UI (entry point)
│   ├── detector.py        # YOLOv8 model loading & drawing
│   ├── recommender.py     # Fertilizer lookup & cost calc
│   ├── predictor.py       # Severity estimation & progression
│   ├── chatbot.py         # Offline rule-based chatbot
│   └── utils.py           # Shared paths/config
├── data/
│   ├── fertilizer_database.csv
│   └── disease_progression_log.csv
├── models/
│   ├── best.pt             # trained YOLOv8 weights (Git LFS)
│   └── last.pt
├── outputs/
│   ├── images/ videos/ reports/
├── scripts/                # standalone offline scripts (optional)
└── requirements.txt
```

---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model weights are present at `models/best.pt`.
   These are tracked with **Git LFS** — after cloning, run:

   ```bash
   git lfs pull
   ```

   to download the actual `.pt` weight files (the repo only stores LFS
   pointers until you do this).

---

## Run Application

```bash
streamlit run app/farm_app.py
```

OR, on Windows, double-click:

```plaintext
run.bat
```

The app opens three tabs:

- **🖼️ Image Detection** — upload a crop image, run detection, see disease,
  severity, predicted progression, and fertilizer recommendation with cost.
- **🎥 Video Detection** — upload a video, get an annotated output video.
- **🤖 Farmer Chatbot** — type in symptoms and get an offline recommendation.

---

## Notes

- `scripts/recommendation_system.py` and
  `scripts/extended_recommendation_system.py` are standalone offline
  scripts used during development/testing. They contain hardcoded local
  Windows paths (`C:\Users\piyus\...`) — update those paths (or point
  them at `models/` and `data/` in this repo) before running them
  directly; they are not required to run the Streamlit app.
