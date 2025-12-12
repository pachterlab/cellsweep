"""Logger Utils"""

import os
import shutil
import yaml
import requests
import subprocess
import logging
from datetime import datetime
from scipy import io, sparse

def my_hello_world():
    print("Hello, world!")

def clear_package_loggers(package_prefix="cellsweep"):
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
        log_file = f"cellsweep_log_{start_time_string}.log"
    
    if log_file:
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        open(log_file, "w").close()  # create or overwrite the log file to ensure it is empty before logging starts. This prevents appending to an existing log file from previous runs, which could lead to confusion when analyzing logs.
        # if os.path.exists(log_file):
        #     raise FileExistsError(f"Log file {log_file} already exists. Please choose a different log file name.")
        print(f"Logging to {log_file}")

    clear_package_loggers(package_prefix="cellsweep")
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

def load_dataset_yaml(yaml_file=None):
    """
    Load dataset YAML from local notebooks/config/.  
    If missing (e.g., running in Colab), download from GitHub.
    """

    dataset_name = os.path.splitext(os.path.basename(yaml_file))[0]

    # URL to raw file on GitHub
    github_dir_url = f"https://raw.githubusercontent.com/pachterlab/cellsweep/main/notebooks/config"
    github_url = f"{github_dir_url}/{dataset_name}.yaml"

    # --- Case 1: Local file exists ---
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)

    # --- Case 2: Download from GitHub ---
    print(f"Config not found locally. Downloading from GitHub:\n{github_url}")

    r = requests.get(github_url)
    if r.status_code != 200:
        try:
            yaml_entries = list_github_yaml_files()
            yaml_entries_message = f"Available configs:\n"
            for yaml_entry in yaml_entries:
                yaml_entries_message += f"Name: {yaml_entry['name']}\nDescription: {yaml_entry['description']}\nURL: {yaml_entry['url']}\n\n"
        except Exception as e:
            yaml_entries_message = ""

        raise FileNotFoundError(
            f"Config {dataset_name}.yaml not found locally or on GitHub.\n"
            f"HTTP {r.status_code}: {github_url}"
            f"{yaml_entries_message}"
        )

    cfg = yaml.safe_load(r.text)

    # Optional: save downloaded file locally in notebooks/config/
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(yaml_file, "w") as f:
        f.write(r.text)

    print(f"Saved downloaded config to:\n{yaml_file}")

    return cfg

def list_github_yaml_files():
    api_url = "https://api.github.com/repos/pachterlab/cellsweep/contents/notebooks/config"

    r = requests.get(api_url)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {api_url}")

    files = r.json()

    yaml_entries = []
    for f in files:
        if f["name"].lower().endswith(".yaml"):
            yaml_url = f["download_url"]

            # Fetch YAML
            yml_text = requests.get(yaml_url).text
            yml = yaml.safe_load(yml_text)

            description = yml.get("description", "(No description provided)")

            yaml_entries.append({
                "name": f["name"],
                "description": description,
                "url": yaml_url,
            })

    return yaml_entries

