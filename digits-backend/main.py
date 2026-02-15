import os
import uuid
import time
import base64
import asyncio
import random
import re
import shutil
from io import BytesIO
from typing import List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from Model_Handler_PTH import Inference_Handler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "../Model_Handler_PTH/Model_PTH"
HISTORY_DIR = "history"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

jobs_db: Dict[str, dict] = {}

class ModelDetail(BaseModel):
    id: str
    name: str

class ImageRequest(BaseModel):
    image: str
    models: list[ModelDetail]

@app.get("/models")
def get_available_models():
    try:
        files = os.listdir(MODELS_DIR)
        
        models_list = []
        for f in files:
            if f.endswith(".pth"):
                clean_name = f.rsplit('.', 1)[0]
                model_id = clean_name.split('_')[1]
                
                models_list.append({
                    "id": model_id,
                    "name": clean_name
                })
        
        return models_list

    except Exception as e:
        print(f"Error scanning models directory: {e}")
        return []

def save_image_to_disk(image_data: str, job_id: str):

    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(BytesIO(image_bytes))
        
        file_path = os.path.join(HISTORY_DIR, f"{job_id}.png")
        image.save(file_path)
        
        return file_path
    
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def process_image_task(job_id: str, models: List[ModelDetail], file_path: str):

    print(f"Start processing job {job_id}")
    results = {}
    
    for model in models:
        full_model_path = os.path.join(MODELS_DIR, f"{model.name}.pth")
        
        if not os.path.exists(full_model_path):
            files = os.listdir(MODELS_DIR)
            for f in files:
                if f.startswith(model.name) or model.id in f:
                    full_model_path = os.path.join(MODELS_DIR, f)
                    break

        if os.path.exists(full_model_path):
            print(f"Running inference with: {model.name}")
            prediction_result = Inference_Handler.predict_digit_real(
                image_path=file_path,
                model_path=full_model_path,
                model_id=model.id
            )
            \
            results[model.name] = prediction_result 
        else:
            results[model.name] = {
                "digit": "?",
                "confidence": "0%",
                "details": f"Model file '{model.name}' not found."
            }
    
    if job_id in jobs_db:
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["results"] = results
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()
        
    print(f"End processing job {job_id}")

@app.post("/submit_job")
async def submit_job(request: ImageRequest, background_tasks: BackgroundTasks):
    
    job_id = str(uuid.uuid4())
    
    file_path = save_image_to_disk(request.image, job_id)
    
    jobs_db[job_id] = {
        "id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "models": request.models,
        "file_path": file_path
    }
    
    background_tasks.add_task(process_image_task, job_id, request.models, file_path)
    
    return {"job_id": job_id, "message": "Job submitted successfully"}

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]

@app.get("/")
def read_root():
    return {"message": "Digits API is running!"}