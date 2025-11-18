"""Logger Utils"""

import os
import shutil
import subprocess
import logging
from datetime import datetime
from scipy import io, sparse

def my_hello_world():
    print("Hello, world!")

def clear_package_loggers(package_prefix="cellmender"):
    for name, obj in logging.Logger.manager.loggerDict.items():
        if name.startswith(package_prefix) and isinstance(obj, logging.Logger):
            obj.handlers.clear()
            obj.propagate = False

def setup_logger(log_file = None, log_level = None, verbose = 0, quiet = False):    
    if log_level is None:
        if quiet or verbose < -1:  # -q
            log_level = logging.CRITICAL
        elif verbose == -1:
            log_level = logging.ERROR
        elif verbose == 0:  # no -q/-v
            log_level = logging.WARNING
        elif verbose == 1:  # -v (and not -q)
            log_level = logging.INFO
        elif verbose >= 2:  # -vv (and not -q)
            log_level = logging.DEBUG
        else:
            raise ValueError(f"Invalid verbose level {verbose}. Use -q for quiet, -v for verbose, and -vv for very verbose.")
    
    if log_file is True:
        # default log file name with timestamp
        start_time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"cellmender_log_{start_time_string}.log"
    
    if log_file:
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        open(log_file, "w").close()  # create or overwrite the log file to ensure it is empty before logging starts. This prevents appending to an existing log file from previous runs, which could lead to confusion when analyzing logs.
        # if os.path.exists(log_file):
        #     raise FileExistsError(f"Log file {log_file} already exists. Please choose a different log file name.")
        print(f"Logging to {log_file}")

    clear_package_loggers(package_prefix="cellmender")
    logger = logging.getLogger(__name__)
    if logger.handlers:  # and repr(logger.handlers[0]) != "<StreamHandler stderr (NOTSET)>":
        return logger
    
    logger.propagate = False
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger