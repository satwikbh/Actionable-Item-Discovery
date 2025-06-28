import unittest
from sklearn.metrics import confusion_matrix, accuracy_score
from Models.Metrics import Metrics

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.metrics_instance = Metrics()

    def test_metrics_calculation(self):
        y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_predicted = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1]

        acc_score, cr_report, cnf_matrix = self.metrics_instance.metrics(y_true, y_predicted)

        # Assert accuracy score
        expected_accuracy = accuracy_score(y_true, y_predicted)
        self.assertAlmostEqual(acc_score, expected_accuracy)

        # Assert classification report (check if it's a string and contains expected keywords)
        self.assertIsInstance(cr_report, str)
        self.assertIn("precision", cr_report)
        self.assertIn("recall", cr_report)
        self.assertIn("f1-score", cr_report)

        # Assert confusion matrix shape and type
        expected_confusion_matrix = confusion_matrix(y_true, y_predicted)
        self.assertEqual(cnf_matrix.shape, expected_confusion_matrix.shape)
        self.assertTrue((cnf_matrix == expected_confusion_matrix).all())

    def test_metrics_with_all_correct_predictions(self):
        y_true = [0, 1, 0, 1]
        y_predicted = [0, 1, 0, 1]

        acc_score, cr_report, cnf_matrix = self.metrics_instance.metrics(y_true, y_predicted)

        self.assertAlmostEqual(acc_score, 1.0)
        self.assertIn("1.00", cr_report) # Check for perfect scores in report
        self.assertTrue((cnf_matrix == [[2, 0], [0, 2]]).all())

    def test_metrics_with_all_incorrect_predictions(self):
        y_true = [0, 1, 0, 1]
        y_predicted = [1, 0, 1, 0]

        acc_score, cr_report, cnf_matrix = self.metrics_instance.metrics(y_true, y_predicted)

        self.assertAlmostEqual(acc_score, 0.0)
        self.assertIn("0.00", cr_report) # Check for zero scores in report
        self.assertTrue((cnf_matrix == [[0, 2], [2, 0]]).all())

if __name__ == '__main__':
    unittest.main()