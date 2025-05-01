# src/utils/logging.py
import logging
import time
from contextlib import contextmanager

def get_logger(name):
    """
    Create a logger with the given name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

@contextmanager
def Timer(name="block"):
    """
    Context manager for timing a code block.
    """
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] elapsed: {end - start:.2f} seconds")
