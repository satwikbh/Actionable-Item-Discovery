import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the sys.path to allow imports from Core and Models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.MLP import MLP
import numpy as np

class TestMLP(unittest.TestCase):
    def setUp(self):
        # Mock ConfigUtil.get_config_instance before initializing MLP
        with patch('Utils.ConfigUtil.ConfigUtil.get_config_instance') as mock_get_config_instance:
            self.mlp = MLP()
            self.mlp.config = mock_get_config_instance.return_value
            self.mlp.config.__getitem__.return_value = "/dummy/path"

        # Mock other dependencies
        self.mlp.log = MagicMock()
        self.mlp.helper = MagicMock()
        self.mlp.metrics = MagicMock()

        # Dummy data for testing
        self.x_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([0, 1, 0])
        self.x_test = np.array([[7, 8], [9, 10]])
        self.y_test = np.array([1, 0])
        self.y_pred = np.array([1, 0])

    def test_train_model(self):
        with patch('Models.MLP.MLPClassifier') as MockMLPClassifier:
            mock_model_instance = MockMLPClassifier.return_value
            mock_model_instance.fit.return_value = mock_model_instance

            model = self.mlp.train_model(self.x_train, self.y_train)

            MockMLPClassifier.assert_called_once()
            mock_model_instance.fit.assert_called_once_with(self.x_train, self.y_train)
            self.assertIsInstance(model, MagicMock) # It's a mock, so check for MagicMock
            self.assertEqual(model, mock_model_instance)

    def test_test_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = self.y_pred

        predictions = self.mlp.test_model(mock_model, self.x_test)

        mock_model.predict.assert_called_once_with(self.x_test)
        np.testing.assert_array_equal(predictions, self.y_pred)

    def test_main(self):
        # Mock train_model and test_model within the MLP instance
        with patch.object(self.mlp, 'train_model', return_value=MagicMock()) as mock_train_model, \
             patch.object(self.mlp, 'test_model', return_value=self.y_pred) as mock_test_model:

            # Mock metrics.metrics return values
            mock_acc_score = 0.95
            mock_cr_report = "classification report"
            mock_cnf_matrix = np.array([[1, 0], [0, 1]])
            self.mlp.metrics.metrics.return_value = (mock_acc_score, mock_cr_report, mock_cnf_matrix)

            result = self.mlp.main(self.x_train, self.x_test, self.y_train, self.y_test)

            mock_train_model.assert_called_once_with(self.x_train, self.y_train)
            mock_test_model.assert_called_once_with(mock_train_model.return_value, self.x_test)
            self.mlp.metrics.metrics.assert_called_once_with(y_true=self.y_test, y_predicted=self.y_pred)
            self.mlp.helper.plot_save_cnf_matrix.assert_called_once_with(
                mock_cnf_matrix,
                flag="test",
                model_name=self.mlp.__class__.__name__,
                image_path="/dummy/path"
            )

            self.assertIn("model", result)
            self.assertIn("metrics", result)
            self.assertEqual(result["metrics"]["accuracy"], mock_acc_score)
            self.assertEqual(result["metrics"]["classification_report"], mock_cr_report)
            np.testing.assert_array_equal(result["metrics"]["confusion_matrix"], mock_cnf_matrix)

if __name__ == '__main__':
    unittest.main()
