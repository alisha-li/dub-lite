from celery import Celery
import sys
import os
import json
from pathlib import Path
import shutil
import logging
import redis

sys.path.append(str(Path(__file__).parent.parent / 'pipeline'))
from main import YTDubPipeline
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Redis client for writing progress so the API can read it (Celery backend may not persist PROGRESS)
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

celery_app = Celery(
    'video_dubbing',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True
)

@celery_app.task(bind=True)
def process_video(self, job_id: str, source_path: str, target_language: str, gemini_api: str = None, gemini_model: str = None, pyannote_key: str = None, groq_api: str = None, groq_model: str = None, hf_token: str = None):
    """Process video in background using Celery"""
    logger.info(f"Starting job {job_id}")
    
    try:
        # If no keys provided, use host's Hugging Face token from env (Helsinki translation + diarization)
        if not any([(gemini_api or "").strip(), (groq_api or "").strip(), (hf_token or "").strip()]):
            hf_token = (os.getenv("HF_TOKEN") or "").strip() or None
            if hf_token:
                logger.info("Using host default: HF token from env (Helsinki translation + diarization)")
            else:
                logger.warning("No API keys provided and HF_TOKEN not set; diarization/translation may fail")
        # If user left HF token blank but provided other keys, still try host HF for diarization (pyannote needs it)
        elif not (hf_token or "").strip():
            hf_token = (os.getenv("HF_TOKEN") or "").strip() or None
            if hf_token:
                logger.info("Using host HF token from env for diarization")

        # Initialize pipeline and run with progress reporting
        pipeline = YTDubPipeline()

        def progress_cb(stage: str, percent: int):
            self.update_state(state='PROGRESS', meta={'stage': stage, 'percent': percent})
            # Also write to Redis so API can read progress (Celery backend may not persist PROGRESS)
            try:
                key = f"job:{job_id}"
                raw = redis_client.get(key)
                data = json.loads(raw) if raw else {}
                data["progress"] = percent
                data["stage"] = stage
                redis_client.set(key, json.dumps(data))
                redis_client.expire(key, 60 * 60 * 24 * 5)
            except Exception as e:
                logger.debug(f"Progress Redis write failed: {e}")

        output_path = pipeline.dub(
            src=source_path,
            targ=target_language,
            hf_token=hf_token,
            gemini_api=gemini_api,
            gemini_model=gemini_model,
            pyannote_key=pyannote_key,
            groq_api=groq_api,
            groq_model=groq_model,
            speakerTurnsPkl=False,
            segmentsPkl=False,
            finalSentencesPkl=False,
            progress_callback=progress_cb
        )
        
        # Copy output to api/outputs directory
        final_output = f"outputs/{job_id}_output.mp4"
        shutil.copy(output_path, final_output)
        
        logger.info(f"Job {job_id} completed! Output: {final_output}")
        
        return {
            "status": "completed",
            "output_path": final_output
        }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
