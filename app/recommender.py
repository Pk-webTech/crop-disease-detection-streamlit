import pandas as pd


def load_recommendations(csv_path):
    df = pd.read_csv(csv_path).fillna("")

    recommendations = {}

    for _, row in df.iterrows():
        disease = str(row["disease"]).strip()

        recommendations[disease] = {
            "fertilizer_name": str(row.get("fertilizer_name", "")).strip(),
            "type": str(row.get("type", "")).strip(),
            "dosage": str(row.get("dosage", "")).strip(),
            "brand": str(row.get("brand", "")).strip(),
            "price_per_litre": float(row.get("price_per_litre", 0))
        }

    return recommendations


def calculate_cost(recommendation, area_m2=100):
    try:
        price = float(recommendation.get("price_per_litre", 0))

        dosage_text = recommendation.get("dosage", "")

        dosage_number = "".join(
            ch for ch in dosage_text
            if ch.isdigit() or ch == "."
        )

        dosage_value = float(dosage_number) if dosage_number else 0

        total_cost = dosage_value * price * (area_m2 / 100)

        return round(total_cost, 2)

    except Exception:
        return 0.0