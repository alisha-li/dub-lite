from celery import Celery
import sys
import os
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).parent.parent / 'pipeline'))
from main import YTDubPipeline
from dotenv import load_dotenv
load_dotenv()

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
def process_video(self, job_id: str, source_path: str, target_language: str):
    """Process video in background using Celery"""
    print(f"Starting job {job_id}")
    
    try:
        # Initialize pipeline
        pipeline = YTDubPipeline()
        
        # Run dubbing
        output_path = pipeline.dub(
            src=source_path,
            targ=target_language,
            hf_token=os.getenv('HF_TOKEN'), # will add drop downs later
            pyannote_key=os.getenv('PYANNOTE_API_KEY'),
            gemini_api=os.getenv('GEMINI_API_KEY'),
            gemini_model="gemini-2.5-flash-lite",
            speakerTurnsPkl=False,
            segmentsPkl=False,
            finalSentencesPkl=False
        )
        
        # Copy output to api/outputs directory
        final_output = f"outputs/{job_id}_output.mp4"
        shutil.copy(output_path, final_output)
        
        print(f"Job {job_id} completed! Output: {final_output}")
        
        return {
            "status": "completed",
            "output_path": final_output
        }
        
    except Exception as e:
        print(f"Job {job_id} failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
