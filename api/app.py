from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import uuid
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
from pydantic import BaseModel
from typing import Optional
import os
import json
import redis
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
# from worker import celery_app, process_video
from fastapi.middleware.cors import CORSMiddleware
import modal
from modal import Function
from modal import Dict

progress_dict = modal.Dict.from_name("dub-lite-progress", create_if_missing=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://dub-lite.alishali.info",
        "https://dub-lite.alishali.info",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def _get_spaces_client():
    """Returns a boto3 S3 client configured for DigitalOcean Spaces."""
    key = os.environ.get("SPACES_ACCESS_KEY")
    secret = os.environ.get("SPACES_SECRET_KEY")
    endpoint = os.environ.get("SPACES_ENDPOINT")
    region = os.environ.get("SPACES_REGION")
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )

@app.post("/api/upload-url")
async def get_upload_url(filename: Optional[str] = Form("video.mp4")):
    """Returns a presigned PUT URL for direct upload to Spaces, plus the object key."""
    client = _get_spaces_client()
    if not client:
        raise HTTPException(status_code=503, detail="Spaces not configured")
    bucket = os.environ.get("SPACES_BUCKET")
    if not bucket:
        raise HTTPException(status_code=503, detail="SPACES_BUCKET not set")
    safe_name = os.path.basename(filename or "video.mp4").replace("..", "")
    if not safe_name.lower().endswith((".mp4", ".webm", ".mkv", ".mov", ".avi")):
        safe_name = f"{safe_name}.mp4" if "." not in safe_name else "video.mp4"
    object_key = f"uploads/{uuid.uuid4()}/{safe_name}"
    try:
        url = client.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": object_key, "ContentType": "video/mp4"},
            ExpiresIn=3600,
        )
    except ClientError as e:
        logger.exception("Failed to generate presigned URL")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"upload_url": url, "object_key": object_key}


@app.get("/")
def read_root():
    return {"message": "Video Dubbing API"}


def _get_spaces_presigned_get_url(object_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL so Modal can download the object from Spaces."""
    client = _get_spaces_client()
    if not client:
        raise HTTPException(status_code=503, detail="Spaces not configured")
    bucket = os.environ.get("SPACES_BUCKET")
    if not bucket:
        raise HTTPException(status_code=503, detail="SPACES_BUCKET not set")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_in,
    )


@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    spaces_object_key: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    target_language: str = Form(None),
    gemini_api: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    pyannote_key: Optional[str] = Form(None),
    groq_api: Optional[str] = Form(None),
    groq_model: Optional[str] = Form(None),
    mistral_api: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None)
):
    job_id = str(uuid.uuid4())

    if spaces_object_key and spaces_object_key.strip():
        src = _get_spaces_presigned_get_url(spaces_object_key.strip())
        source_path = spaces_object_key
        source_type = "spaces"
    elif file:
        # Upload directly to Spaces, then pass the URL to Modal
        client = _get_spaces_client()
        bucket = os.environ.get("SPACES_BUCKET")
        if not client or not bucket:
            raise HTTPException(status_code=503, detail="Spaces not configured")
        object_key = f"uploads/{job_id}/{file.filename}"
        file_bytes = await file.read()
        client.put_object(Bucket=bucket, Key=object_key, Body=file_bytes, ContentType="video/mp4")
        src = _get_spaces_presigned_get_url(object_key)
        source_path = object_key
        source_type = "upload"
    else:
        raise HTTPException(status_code=400, detail="Either file or spaces_object_key is required")

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

    mistral_api_val = (mistral_api or "").strip() or os.environ.get("MISTRAL_API_KEY")

    run_pipeline = modal.Function.from_name("dub-lite", "run_dubbing_pipeline")
    call = run_pipeline.spawn(
        job_id=job_id,
        src=src,
        targ=target_language,
        hf_token=hf_token,
        pyannote_key=pyannote_key,
        gemini_api=gemini_api,
        groq_api=groq_api,
        groq_model=groq_model,
        gemini_model=gemini_model,
        mistral_api=mistral_api_val,
    )

    job_data["modal_call_id"] = call.object_id
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

    if job.get("status") in ["completed", "failed"]:
        resp = {
            "job_id": job_id,
            "status": job["status"],
            "error": job.get("error"),
        }
        if job.get("output_spaces_key"):
            resp["output_url"] = _get_spaces_presigned_get_url(job["output_spaces_key"], expires_in=86400)
        return resp

    modal_call_id = job["modal_call_id"]
    if not modal_call_id:
        return {"job_id": job_id, "status": "unknown"}

    try:
        # Get the call and check if finished
        call = modal.FunctionCall.from_id(modal_call_id)
        
        try:
            output_key = call.get(timeout=0)
            
            job["status"] = "completed"
            job["output_spaces_key"] = output_key
            redis_client.set(f"job:{job_id}", json.dumps(job))
            redis_client.expire(f"job:{job_id}", 60 * 60 * 24 * 5)
            
        except TimeoutError:
            # Still running — read live progress from Modal Dict
            job["status"] = "processing"
            try:
                prog = progress_dict.get(job_id)
                if prog:
                    job["stage"] = prog.get("stage", "Pipeline running on GPU...")
                    job["progress"] = prog.get("progress", 0)
                    logger.info("Job %s progress: %s%% - %s", job_id, job["progress"], job["stage"])
                else:
                    job["stage"] = "Pipeline running on GPU..."
                    job["progress"] = 0
            except Exception as e:
                logger.warning("Progress dict read failed for job %s: %s", job_id, e)
                job["stage"] = "Pipeline running on GPU..."
                job["progress"] = 0
    
    except Exception as e:
        # Failed or not found — full traceback goes to the uvicorn terminal
        job["status"] = "failed"
        job["error"] = str(e)
        redis_client.set(f"job:{job_id}", json.dumps(job))
        logger.exception("Modal job %s failed", job_id)
    
    # 3. Return current status
    resp = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "error": job.get("error"),
    }
    if job.get("output_spaces_key"):
        resp["output_url"] = _get_spaces_presigned_get_url(job["output_spaces_key"], expires_in=86400)
    return resp


    # task = celery_app.AsyncResult(job["celery_task_id"])

    # # job["status"] = task.state
    # if task.state == "SUCCESS":
    #     result = task.result
    #     if result.get("status") == "failed":
    #         job["status"] = "failed"
    #         job["error"] = result.get("error", "Unknown error")
    #     elif result.get("status") == "completed":
    #         job["status"] = "completed"
    #         job["output_path"] = result.get("output_path")
    #         job["completed_at"] = datetime.now().isoformat()
    # elif task.state == "FAILURE":
    #     job["status"] = "failed"
    #     job["error"] = str(task.info)
    # else:
    #     # PENDING, STARTED, or PROGRESS — re-fetch job from Redis for latest progress (worker writes there)
    #     job["status"] = "processing"
    #     raw_latest = redis_client.get(f"job:{job_id}")
    #     if raw_latest:
    #         latest = json.loads(raw_latest)
    #         job["progress"] = latest.get("progress", 0)
    #         job["stage"] = latest.get("stage", "Starting...")
    #     else:
    #         job["progress"] = job.get("progress", 0)
    #         job["stage"] = job.get("stage", "Starting...")

    # return job
    
@app.get("/api/jobs/{job_id}/download")
def download_video(job_id: str):
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(job_data)
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Video not ready yet.")

    output_key = job.get("output_spaces_key")
    if not output_key:
        raise HTTPException(status_code=404, detail="Output file not found")

    url = _get_spaces_presigned_get_url(output_key, expires_in=3600)
    return RedirectResponse(url=url, status_code=302)

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job_data = redis_client.get(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = json.loads(job_data)

    try:
        client = _get_spaces_client()
        bucket = os.environ.get("SPACES_BUCKET")
        if client and bucket:
            for key_field in ("source_path", "output_spaces_key"):
                obj_key = job.get(key_field)
                if obj_key and obj_key.startswith(("uploads/", "outputs/")):
                    client.delete_object(Bucket=bucket, Key=obj_key)
    except Exception as e:
        logger.warning("Error deleting Spaces objects for job %s: %s", job_id, e)
    
    redis_client.delete(f"job:{job_id}")
    
    return {"message": "Job deleted successfully"}