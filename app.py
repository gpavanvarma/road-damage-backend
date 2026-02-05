from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

# -------------------------
# App initialization
# -------------------------
app = FastAPI(title="Road Damage Detection API")

# Allow frontend (Netlify) to access backend (Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load YOLOv8 model
# -------------------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Create output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Root endpoint (health check)
# -------------------------
@app.get("/")
def root():
    return {"status": "Road Damage Detection API is running"}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file,
    runs YOLOv8 inference,
    draws bounding boxes,
    returns processed image name
    """

    # Read uploaded image
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img)[0]

    # Draw bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save output image
    image_name = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, img)

    return {
        "message": "Detection successful",
        "image_name": image_name
    }

# -------------------------
# Serve output images
# -------------------------
@app.get("/outputs/{image_name}")
def get_output_image(image_name: str):
    image_path = os.path.join(OUTPUT_DIR, image_name)

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    return FileResponse(image_path, media_type="image/jpeg")
