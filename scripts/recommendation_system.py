from ultralytics import YOLO
import pandas as pd
import sys
import os

# Redirect output to text file
sys.stdout = open(r"C:\Users\piyus\Desktop\result_summary.txt", "w", encoding="utf-8")

# 🧩 1️⃣ Load the trained model
model = YOLO(r"C:\Users\piyus\runs\detect\combined_v4_yolov8s_fixed\weights\best.pt")

# 🧩 2️⃣ Load fertilizer database
fertilizer_db = pd.read_csv(r"C:\Users\piyus\Desktop\fertilizer_database.csv")

# 🧩 3️⃣ Create recommendation lookup dictionary
recommendations = {
    row["disease"]: {
        "fertilizer_name": row["fertilizer_name"],
        "type": row["type"],
        "dosage": row["dosage"],
        "brand": row["brand"]
    }
    for _, row in fertilizer_db.iterrows()
}

# 🧩 4️⃣ Run detection
results = model.predict(
    source=r"C:\Users\piyus\runs\detect\combined_v4_yolov8s_fixed\testimages and video folder\AI_Plant_Disease_Detection_Video_Generation.mp4",  # Change path as needed
    conf=0.5,
    show=True
)

# 🧩 5️⃣ Display and save detected results + recommendations
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
print("✅ Report saved successfully at: C:\\Users\\piyus\\Desktop\\result_summary.txt")

# Close the file output stream
sys.stdout.close()


