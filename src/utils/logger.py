import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'BoVWPipeline',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = 'logs'
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name.lower()}_{timestamp}.log'
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'BoVWPipeline') -> logging.Logger:
    """
    Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)



logger = get_logger('BoVW Pipeline')