from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from inference_sdk import InferenceHTTPClient

# Roboflow Client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dfc35RC7Kqz9RwyiMbNC"
)

app = FastAPI()

# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Roboflow Eye Detection API is up"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    
    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Send to Roboflow
    result = client.infer(temp_path, model_id="entrmalfinal/2")

    # Clean up
    os.remove(temp_path)

    if "predictions" in result and result["predictions"]:
        pred = result["predictions"][0]
        return {
            "prediction": pred["class"],
            "confidence": round(pred["confidence"] * 100, 2)
        }
    else:
        return {"error": "No prediction returned"}
