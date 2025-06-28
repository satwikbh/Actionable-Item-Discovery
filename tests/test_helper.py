import unittest
import sys
import os
from unittest.mock import patch, mock_open

# Add the project root to the sys.path to allow imports from Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.Helper import Helper

class TestHelper(unittest.TestCase):

    def test_is_list_not_empty(self):
        self.assertTrue(Helper.is_list_not_empty([1, 2, 3]))
        self.assertFalse(Helper.is_list_not_empty([]))

    def test_is_list_empty(self):
        self.assertTrue(Helper.is_list_empty([]))
        self.assertFalse(Helper.is_list_empty([1, 2, 3]))

    @patch('Utils.Helper.plt.savefig')
    @patch('Utils.Helper.heatmap')
    @patch('Utils.Helper.set')
    @patch('Utils.Helper.DataFrame')
    @patch('Utils.Helper.plt.figure')
    def test_plot_save_cnf_matrix(self, mock_figure, mock_DataFrame, mock_set, mock_heatmap, mock_savefig):
        cnf_matrix = [[10, 1], [2, 15]]
        model_name = "TestModel"
        flag = "test"
        image_path = "/tmp/images"

        Helper.plot_save_cnf_matrix(cnf_matrix, model_name, flag, image_path)

        
        mock_DataFrame.assert_called_once_with(cnf_matrix, range(2), range(2))
        mock_set.assert_called_once_with(font_scale=1.4)
        mock_heatmap.assert_called_once()
        mock_savefig.assert_called_once_with("/tmp/images/confusion_matrix_TestModel_test.png")

    @patch('builtins.open', new_callable=mock_open, read_data="line1\nline2\nline3")
    def test_load_list_from_file(self, mock_file_open):
        filename = "dummy.txt"
        expected_list = ["line1", "line2", "line3"]
        result = Helper.load_list_from_file(filename)

        mock_file_open.assert_called_once_with(filename, "r")
        self.assertEqual(result, expected_list)

if __name__ == '__main__':
    unittest.main()
