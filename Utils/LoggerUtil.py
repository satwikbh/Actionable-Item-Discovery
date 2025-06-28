import json
import logging
import os
from logging.config import dictConfig


class LoggerUtil(object):
    def __init__(self, default_path):
        path = os.path.join(os.path.dirname(__file__), "logging.json")
        with open(path, "r") as f:
            logging_config = json.load(f)

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        dictConfig(logging_config)
        log = logging.getLogger()
        self._logger = log

    def get(self):
        return self._logger
