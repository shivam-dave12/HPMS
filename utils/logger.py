"""utils/logger.py — Thin wrapper around stdlib logging for HL SDK modules."""
import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
