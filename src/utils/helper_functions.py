"""
Central helper functions and configuration utilities for the GoPredict project.
"""

import os
import yaml
import logging
from pathlib import Path

# --- YAML Loader ---   
def load_yaml_config(config_file: str) -> dict:
    """
    Loads a YAML configuration file and returns a dictionary of its contents.
    """
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
    
def configure_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    modules = ['train', 'predict', 'evaluate', 'api']
    loggers = {}

    for module in modules:
        logger = logging.getLogger(f'gopredict.{module}')
        file_handler = logging.FileHandler(f'{log_dir}/{module}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        loggers[module] = logger
        
    return loggers

def safe_segment_name(label: str) -> str:
    """Convert MSRP segment label to a valid file name."""
    s = str(label)
    return(
        s.replace(' ', '_')
        .replace('(', '').replace(')', '')
        .replace('<', 'lt').replace('>', 'gt')
        .replace('-', '_').replace('/', '_')
        .replace('.', '_').replace(',', '_')
        .replace('=', '_')
    )

