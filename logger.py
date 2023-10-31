"""File used to initialize logging"""
import logging
# Any execution that occurs, error, run time, logging does that
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# os.path.join(in which folder, what name of the new folder, what name of the file in which the log will be written)
logs_path = os.path.join("logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# You always have to set the basic configuration of the logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


