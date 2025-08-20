import io
import logging
import os

import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(
    app, include_in_schema=False, should_gzip=True
)

# Load your best model once on startup
mlruns_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

best_run_id = "065f403e8f514882b45cf1146756efa0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlflow.pytorch.load_model(
    f"runs:/{best_run_id}/model", map_location=device
)
model.to(device)
model.eval()

# Define your preprocessing (must match training)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Garbage Classification API"}


@app.post("/predict")
async def predict(request: Request):
    client_host = request.client.host
    logger.info(f"Received prediction request from {client_host}")

    try:
        image_bytes = await request.body()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        logger.warning(f"Invalid image data from {client_host}")
        raise HTTPException(status_code=400, detail="Invalid image data")
    except Exception as e:
        logger.error(f"Error reading image from {client_host}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error reading image: {e}"
        )

    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            preds = torch.nn.functional.softmax(output, dim=1)
            predicted_class = preds.argmax(dim=1).item()
            confidence = preds.max().item()
        logger.info(
            f"Prediction for {client_host}: "
            f"class={predicted_class}, confidence={confidence:.4f}"
        )
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        logger.error(f"Inference error for {client_host}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
