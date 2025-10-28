import json
import logging
import os


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level)
    if os.environ.get("JSON_LOGS", "0") == "1":
        logging.getLogger().handlers = [JSONLogHandler()]


class JSONLogHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            msg = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                msg["exc_info"] = self.formatException(record.exc_info)
            self.stream.write(json.dumps(msg) + "\n")
        except Exception:
            super().emit(record)

