# logger_config.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standardized logger instance.

    This function ensures that all parts of the application use a consistent
    logging format and level. The logger is configured to stream to stdout.
    It is idempotent; if a logger with the given name has already been
    configured, it will not be re-configured.

    Args:
        name (str): The name for the logger, typically `__name__`.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create a handler to write logs to standard output
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    return logger
