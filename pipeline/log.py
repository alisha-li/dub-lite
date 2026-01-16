import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(names)s - %(levelnames)s- %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )

# utils.py  
import logging
logger = logging.getLogger(__name__)  # Just this!