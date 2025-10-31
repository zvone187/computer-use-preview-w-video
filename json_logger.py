# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present (from extra={"extra_fields": {...}})
        if hasattr(record, "extra_fields"):
            if isinstance(record.extra_fields, dict):
                log_obj.update(record.extra_fields)
            else:
                log_obj["extra_fields"] = record.extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


def setup_json_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger configured for JSON output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler with JSON formatter
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


def get_json_logger(name: str = __name__) -> logging.Logger:
    """Get an existing logger or create a new one with JSON formatting."""
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers yet, set it up
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    elif not isinstance(logger.handlers[0].formatter, JSONFormatter):
        # Replace non-JSON formatter with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.removeHandler(logger.handlers[0])
        logger.addHandler(handler)
    
    return logger

