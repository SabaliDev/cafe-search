import json
import logging
from app.config import settings


def setup_logging() -> None:
    logging.basicConfig(level=settings.log_level)


def log_event(event: str, payload: dict) -> None:
    logger = logging.getLogger("cafe-search")
    data = {"event": event, **payload}
    logger.info(json.dumps(data))
