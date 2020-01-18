import json
import logging
import os
from logging.config import dictConfig


class LoggerUtil(object):
    def __init__(self, default_path):
        path = os.path.dirname(__file__) + "/" + "logging.json"
        logging_config = json.load(open(path))
        dictConfig(logging_config)
        log = logging.getLogger()
        self._logger = log

    def get(self):
        return self._logger
