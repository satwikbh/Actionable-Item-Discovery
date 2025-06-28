import unittest
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.ConfigUtil import ConfigUtil

class TestConfigUtil(unittest.TestCase):

    def setUp(self):
        # Create a dummy Config.json for testing
        self.test_config_path = 'Config.json'
        self.original_config_content = {
            "data_path": "./",
            "chunksize": 1000,
            "image_path": "./Images/",
            "tagged_path": "./",
            "action_verbs_path": "./",
            "models_path": "./SavedModels/",
            "n_partitions": 10
        }
        with open(self.test_config_path, 'w') as f:
            json.dump(self.original_config_content, f)

    def tearDown(self):
        # Clean up the dummy Config.json
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_get_config_instance(self):
        config = ConfigUtil.get_config_instance()
        self.assertIsInstance(config, dict)
        self.assertEqual(config, self.original_config_content)
        self.assertEqual(config['data_path'], './')
        self.assertEqual(config['image_path'], './Images/')

    def test_get_config_instance_missing_file(self):
        # Remove the dummy Config.json to simulate a missing file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

        with self.assertRaises(FileNotFoundError):
            ConfigUtil.get_config_instance()

if __name__ == '__main__':
    unittest.main()