import logging

# Configure logging once
logging.basicConfig(
    level=logging.DEBUG,  # Adjust level as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rp.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

# Create a logger instance for use in other modules
logger = logging.getLogger("RP_Logger")
