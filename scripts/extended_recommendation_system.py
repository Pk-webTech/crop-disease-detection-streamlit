# ================================================================
# 🌾 AI-Based Crop Disease Detection & Fertilizer Recommendation System
# Author: Piyush Kumar
# Faculty: Navami T M
# Features:
#  1. Offline YOLOv8 Detection
#  2. Fertilizer Recommendation (CSV)
#  3. Smart Cost Estimation
#  4. Disease Progression Prediction
#  5. Offline Chatbot for Farmer Guidance
# ================================================================

from pathlib import Path
from datetime import datetime
import os

from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
CSV_PATH = BASE_DIR / "data" / "fertilizer_database.csv"
LOG_PATH = BASE_DIR / "data" / "disease_progression_log.csv"

# Update this to the image you want to analyze
SOURCE_IMAGE = BASE_DIR / "samples" / "images" / "sample.jpg"

# ================================================================
# Load Trained Model
# ================================================================
model = YOLO(str(MODEL_PATH))

# ================================================================
# Load Fertilizer Database
# ================================================================
fertilizer_db = pd.read_csv(CSV_PATH)

# Convert CSV into a dictionary for quick lookup
recommendations = {}
for _, row in fertilizer_db.iterrows():
    recommendations[row["disease"]] = {
        "fertilizer_name": row["fertilizer_name"],
        "type": row["type"],
        "dosage": row["dosage"],
        "brand": row["brand"],
        "price_per_litre": row.get("price_per_litre", 0)
    }

# ================================================================
# Disease Detection
# ================================================================
print("\n🧠 Running YOLOv8 Model for Detection...\n")
results = model.predict(
    source=str(SOURCE_IMAGE),
    conf=0.5,
    save=True,
    show=False
)

# ================================================================
# Severity Calculation (Simulated based on bounding box coverage)
# ================================================================
def calculate_severity(result):
    """Estimate infection severity percentage based on detected area."""
    total_area = 1.0  # Simulated reference area (normalized)
    if len(result.boxes.xywh) == 0:
        return 0
    infected_area = np.sum(result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]) / total_area
    return min(round(infected_area * 100, 2), 100)

# ================================================================
# Log and Predict Disease Progression
# ================================================================
def update_progression_log(disease, severity):
    """Append severity data to CSV and train regression for prediction."""
    entry = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease": disease,
        "severity_percent": severity
    }])

    if os.path.exists(LOG_PATH):
        entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
        data = pd.read_csv(LOG_PATH, names=["date", "disease", "severity_percent"])
    else:
        entry.to_csv(LOG_PATH, index=False)
        data = entry

    # Filter specific disease history
    data = data[data["disease"] == disease]

    if len(data) >= 3:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data["severity_percent"].values
        regression_model = LinearRegression().fit(X, y)
        future = np.array([[len(data) + i] for i in range(1, 4)])
        prediction = regression_model.predict(future)
        increase = round(prediction[-1] - y[-1], 2)
        print(f"📈 Predicted infection increase for '{disease}': +{increase}% in 3 days.\n")
    else:
        print(f"📊 Disease history too short to predict progression for '{disease}'.\n")

# ================================================================
# Display Recommendations + Cost + Prediction
# ================================================================
for r in results:
    for c in r.boxes.cls:
        label = model.names[int(c)]
        severity = calculate_severity(r)

        print(f"🩺 Detected Disease: {label}")
        print(f"📉 Estimated Infection Severity: {severity}%")

        if label in recommendations:
            rec = recommendations[label]
            price = rec["price_per_litre"]
            total_cost = round(price * 2.5, 2)  # Simulated cost per acre

            print(f"💊 Recommended Fertilizer: {rec['fertilizer_name']} ({rec['type']})")
            print(f"📦 Brand: {rec['brand']}")
            print(f"💧 Dosage: {rec['dosage']}")
            print(f"💰 Estimated Treatment Cost: ₹{total_cost}")
            print("-" * 60)

            # Log severity and predict future progression
            update_progression_log(label, severity)
        else:
            print(f"⚠️ No fertilizer recommendation found for {label}\n")

# ================================================================
# Offline Chatbot for Farmer Guidance
# ================================================================
def chatbot_response(user_input):
    user_input = user_input.lower()
    for _, row in fertilizer_db.iterrows():
        if row["disease"].lower() in user_input:
            return (f"This may indicate {row['disease']}. "
                    f"Apply {row['fertilizer_name']} ({row['dosage']}) by {row['brand']}.")
    return "Sorry, I don't have information on that symptom."

if __name__ == "__main__":
    print("\n🤖 Farmer Chatbot is active! (Type 'exit' to quit)")
    while True:
        query = input("Farmer: ")
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye! Stay healthy 🌾")
            break
        print("Chatbot:", chatbot_response(query))
