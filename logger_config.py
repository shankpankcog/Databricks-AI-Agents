# logger_config.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standardized logger instance.

    Args:
        name (str): The name for the logger, typically `__name__`.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
