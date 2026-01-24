from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import os
import sys
from worker import celery_app, process_videos

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

jobs = {}

class JobRequest(BaseModel):
    source_type: str
    source_url: Optional[str] = None
    target_language: str

@app.get("/")
def read_root():
    return {"message": "Video Dubbing API"}

@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    source_url: Optional[str] = None,
    target_language: Optional[str] = None
):
    job_id = str(uuid.uuid4())

    if file: 
        file_path = f"uploads/{job_id}_{file.filename}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "source_type": "upload",
            "source_url": file_path,
            "target_language": target_language,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    else:
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "source_type": "youtube",
            "source_url": source_url,
            "target_language": target_language,
            "created_at": datetime.now().isoformat()
        }

    jobs[job_id] = job_data

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job created successfully"
    }

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/download")
def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video not ready yet")
    
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4"
    )