import unittest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

# Add the project root to the sys.path to allow imports from Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.LoggerUtil import LoggerUtil

class TestLoggerUtil(unittest.TestCase):
    @patch('Utils.LoggerUtil.os.path.exists', return_value=True)
    @patch('Utils.LoggerUtil.os.makedirs')
    @patch('Utils.LoggerUtil.json.load', return_value={'version': 1, 'disable_existing_loggers': False})
    @patch('Utils.LoggerUtil.dictConfig')
    @patch('Utils.LoggerUtil.logging.getLogger')
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_initialization(self, mock_file_open, mock_getLogger, mock_dictConfig, mock_json_load, mock_makedirs, mock_exists):
        mock_logger_instance = MagicMock()
        mock_getLogger.return_value = mock_logger_instance

        logger_util = LoggerUtil("test_logger")

        # Assertions for __init__
        mock_file_open.assert_called_once_with('/Users/satwik/Documents/Satwik/Actionable-Item-Discovery/Utils/logging.json', 'r')
        mock_json_load.assert_called_once()
        mock_exists.assert_called_once()
        mock_makedirs.assert_not_called() # Should not be called if exists returns True
        mock_dictConfig.assert_called_once_with({'version': 1, 'disable_existing_loggers': False})
        mock_getLogger.assert_called_once()
        self.assertEqual(logger_util.get(), mock_logger_instance)

    @patch('Utils.LoggerUtil.os.path.abspath', return_value='/Users/satwik/Documents/Satwik/Actionable-Item-Discovery/Utils/LoggerUtil.py')
    @patch('Utils.LoggerUtil.os.path.exists', return_value=False)
    @patch('Utils.LoggerUtil.os.makedirs')
    @patch('Utils.LoggerUtil.json.load', return_value={'version': 1, 'disable_existing_loggers': False})
    @patch('Utils.LoggerUtil.dictConfig')
    @patch('Utils.LoggerUtil.logging.getLogger')
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_initialization_create_logs_dir(self, mock_file_open, mock_getLogger, mock_dictConfig, mock_json_load, mock_makedirs, mock_exists, mock_abspath):
        mock_logger_instance = MagicMock()
        mock_getLogger.return_value = mock_logger_instance

        logger_util = LoggerUtil("test_logger")

        # Assertions for __init__ when logs directory does not exist
        mock_exists.assert_called_once()
        mock_makedirs.assert_called_once_with('/Users/satwik/Documents/Satwik/Actionable-Item-Discovery/Utils/../logs')
        mock_dictConfig.assert_called_once_with({'version': 1, 'disable_existing_loggers': False})
        mock_getLogger.assert_called_once()
        self.assertEqual(logger_util.get(), mock_logger_instance)

    @patch('Utils.LoggerUtil.logging.getLogger')
    @patch('Utils.LoggerUtil.dictConfig')
    @patch('Utils.LoggerUtil.json.load', return_value={'version': 1, 'disable_existing_loggers': False})
    @patch('Utils.LoggerUtil.os.makedirs')
    @patch('Utils.LoggerUtil.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_get_method(self, mock_open, mock_exists, mock_makedirs, mock_json_load, mock_dictConfig, mock_getLogger):
        # Test the get method returns the stored logger
        mock_logger_instance = MagicMock()
        mock_getLogger.return_value = mock_logger_instance

        logger_util = LoggerUtil("test_logger")
        self.assertEqual(logger_util.get(), mock_logger_instance)

if __name__ == '__main__':
    unittest.main()
