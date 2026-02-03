from celery import Celery
import sys
import os
from pathlib import Path
import shutil
import logging

sys.path.append(str(Path(__file__).parent.parent / 'pipeline'))
from main import YTDubPipeline
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

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
        # Initialize pipeline
        pipeline = YTDubPipeline()
        
        # Run dubbing
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
            finalSentencesPkl=False
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
