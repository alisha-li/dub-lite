from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import os
import json
import redis
from worker import celery_app, process_video
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# class JobRequest(BaseModel):
#     source_type: str
#     source_url: Optional[str] = None
#     target_language: str

@app.get("/")
def read_root():
    return {"message": "Video Dubbing API"}

@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    source: Optional[str] = Form(None),
    target_language: str = Form(None),
    gemini_api: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    pyannote_key: Optional[str] = Form(None),
    groq_api: Optional[str] = Form(None),
    groq_model: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None)
):
    job_id = str(uuid.uuid4())

    if file: 
        file_path = f"uploads/{job_id}_{file.filename}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        source_path = file_path
        source_type = "upload"

    else:
        source_path = source
        source_type = "youtube"

    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "stage": "Starting...",
        "source_type": source_type,
        "source_path": source_path,
        "target_language": target_language,
        "created_at": datetime.now().isoformat(),
    }

    task = process_video.delay(
        job_id,
        source_path,
        target_language,
        gemini_api,
        gemini_model,
        pyannote_key,
        groq_api,
        groq_model,
        hf_token
    )

    job_data["celery_task_id"] = task.id
    redis_client.set(f"job:{job_id}", json.dumps(job_data))
    redis_client.expire(f"job:{job_id}", 60 * 60 * 24 * 5)  # 5 days

    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Job created successfully"
    }

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job_data = redis_client.get(f"job:{job_id}")
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = json.loads(job_data)
    
    task = celery_app.AsyncResult(job["celery_task_id"])
    job["status"] = task.state
    if task.state == "SUCCESS":
        result = task.result
        if result.get("status") == "failed":
            job["status"] = "failed"
            job["error"] = result.get("error", "Unknown error")
        elif result.get("status") == "completed":
            job["status"] = "completed"
            job["output_path"] = result.get("output_path")
            job["completed_at"] = datetime.now().isoformat()
    elif task.state == "FAILURE":
        job["status"] = "failed"
        job["error"] = str(task.info)
    else:
        # PENDING, STARTED, or PROGRESS — re-fetch job from Redis for latest progress (worker writes there)
        job["status"] = "processing"
        raw_latest = redis_client.get(f"job:{job_id}")
        if raw_latest:
            latest = json.loads(raw_latest)
            job["progress"] = latest.get("progress", 0)
            job["stage"] = latest.get("stage", "Starting...")
        else:
            job["progress"] = job.get("progress", 0)
            job["stage"] = job.get("stage", "Starting...")

    return job
    

@app.get("/api/jobs/{job_id}/download")
def download_video(job_id: str):
    job_data = redis_client.get(f"job:{job_id}")
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(job_data)

    task = celery_app.AsyncResult(job["celery_task_id"])
    if task.state == "SUCCESS":
        result = task.result
        if result.get("status") == "completed":
            output_path = result.get("output_path")
            if not output_path or not os.path.exists(output_path):
                raise HTTPException(status_code=404, detail="Output file not found")
            return FileResponse(
                path=output_path,
                media_type="video/mp4",
                filename=f"dubbed_{job_id}.mp4"
            )
        if result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=f"Job failed: {result.get('error')}")
    if task.state == "FAILURE":
        raise HTTPException(status_code=400, detail=f"Job failed: {task.info}")

    raise HTTPException(status_code=400, detail="Video not ready yet.")

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job_data = redis_client.get(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = json.loads(job_data)

    try:
        if "source_path" in job and os.path.exists(job["source_path"]):
            os.remove(job["source_path"])
        if "output_path" in job and os.path.exists(job["output_path"]):
            os.remove(job["output_path"])
    except Exception as e:
        print(f"Error deleting files: {e}")
    
    redis_client.delete(f"job:{job_id}")
    
    return {"message": "Job deleted successfully"}