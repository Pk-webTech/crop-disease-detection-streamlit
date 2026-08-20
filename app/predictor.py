import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os



def estimate_severity(result):
    if len(result.boxes) == 0:
        return 0.0

    image_height, image_width = result.orig_shape[:2]
    total_area = image_height * image_width

    boxes = result.boxes.xywh.cpu().numpy()

    infected_area = np.sum(boxes[:, 2] * boxes[:, 3])

    severity = (infected_area / total_area) * 1000

    return round(min(severity, 100), 2)



def predict_progression(history):
    if len(history) < 3:
        return None

    X = np.arange(len(history)).reshape(-1, 1)
    y = np.array(history)

    regression_model = LinearRegression()
    regression_model.fit(X, y)

    prediction = regression_model.predict([[len(history)]])[0]

    return round(float(prediction), 2)



def update_progression_log(log_path, disease, severity):
    entry = pd.DataFrame([
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": disease,
            "severity": severity
        }
    ])

    if os.path.exists(log_path):
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)