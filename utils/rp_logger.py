import logging
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Generate unique log filename with timestamp
log_filename = datetime.now().strftime("logs/run_%Y-%m-%d_%H-%M-%S.log")

# Configure logging to show only time
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",  # Only show hour:minute:second
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RP_Logger")