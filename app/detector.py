from ultralytics import YOLO
import cv2



def load_model(model_path):
    return YOLO(model_path)



def detect_image(model, image, confidence_threshold=0.45):
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=640,
        verbose=False
    )

    return results[0]



def draw_detections(
    image,
    boxes,
    classes,
    confidences,
    names,
    recommendations,
    show_overlay=True
):

    for (x1, y1, x2, y2), cls, confidence in zip(
        boxes,
        classes,
        confidences
    ):

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        label = names[int(cls)]

        color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {confidence:.2f}"

        if show_overlay:
            if label in recommendations:
                fertilizer = recommendations[label]["fertilizer_name"]

                if fertilizer:
                    text += f" -> {fertilizer}"

        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            image,
            (x1, y1 - text_height - 8),
            (x1 + text_width + 8, y1),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    return image
