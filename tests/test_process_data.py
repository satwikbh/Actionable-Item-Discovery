import unittest
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.ProcessData import ProcessData


class TestProcessData(unittest.TestCase):
    def setUp(self):
        self.process_data = ProcessData()

    def test_get_regex_pattern(self):
        expected_pattern = (
            r"(?:Message-ID: )([\s\S]*)(?:\n)"
            r"(?:Date: )([\s\S]*)(?:\n)"
            r"(?:From: )(.*)"
            r"(?:(?:(?:\n)(?=(?:To:) )(?:To: )([\s\S]*)(?:\n))|(?:(?:\n)(?!(?:To: ))))"
            r"(?:Subject: )([\s\S]*)(?:\n)"
            r"(?:Mime-Version: )([\s\S]*)(?:\n)"
            r"(?:Content-Type: )([\s\S]*)(?:\n)"
            r"(?:Content-Transfer-Encoding: )([\s\S]*)(?:\n)"
            r"(?:X-From: )([\s\S]*)(?:\n)"
            r"(?:X-To: )([\s\S]*)(?:\n)"
            r"(?:X-cc: )([\s\S]*)(?:\n)"
            r"(?:X-bcc: )([\s\S]*)(?:\n)"
            r"(?:X-Folder: )([\s\S]*)(?:\n)"
            r"(?:X-Origin: )([\s\S]*)(?:\n)"
            r"([\s\S]*$)"
        )
        self.assertEqual(self.process_data.get_regex_pattern(), expected_pattern)

    def test_clean_text_re_subject(self):
        subject = "re: Meeting"
        message = "Hello"
        self.assertEqual(self.process_data.clean_text(subject, message), ["", ""])

    def test_clean_text_forwarded_message(self):
        subject = "Meeting"
        message = "---- forwarded message ----"
        self.assertEqual(self.process_data.clean_text(subject, message), ["", ""])

    def test_clean_text_no_match(self):
        subject = "Meeting"
        message = "Hello"
        self.assertEqual(
            self.process_data.clean_text(subject, message), [subject, message]
        )

    @patch("Core.ProcessData.dd.from_pandas")
    def test_get_sub_message(self, mock_from_pandas):
        # Mock the dask.dataframe.from_pandas to return a pandas DataFrame
        # that behaves like a dask dataframe for the purpose of this test.
        # This is a simplified mock, as the actual regex extraction happens on the pandas Series.
        mock_df = pd.DataFrame(
            {
                "message": [
                    "Message-ID: <1>\nDate: 1\nFrom: a@b.com\nSubject: Test Subject\nMime-Version: 1\nContent-Type: 1\nContent-Transfer-Encoding: 1\nX-From: a\nX-To: b\nX-cc: c\nX-bcc: d\nX-Folder: e\nX-Origin: f\nX-FileName: g\nTest Message Body"
                ]
            }
        )

        # The actual extraction is done by pandas .str.extract, so we need to ensure
        # that the mock_df has a .message.str attribute that can be called with .extract
        mock_df.message = mock_df.message.astype(
            str
        )  # Ensure it's string type for .str accessor

        # We don't need to mock the return of from_pandas for this test,
        # as get_sub_message directly uses the pandas .str.extract method.
        # The mock_from_pandas is just to satisfy the patch decorator.

        result_df = self.process_data.get_sub_message(mock_df)

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn("subject", result_df.columns)
        self.assertIn("message", result_df.columns)
        self.assertEqual(result_df["subject"].iloc[0], "Test Subject")
        self.assertEqual(result_df["message"].iloc[0], "Test Message Body")

    @patch("Core.ProcessData.load")
    @patch("Core.ProcessData.dd.from_pandas")
    @patch("Core.ProcessData.Logic")
    def test_clean_df(self, MockLogic, mock_from_pandas, mock_load):
        # Mock spacy.load
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        # Mock Logic class and its method
        mock_logic_instance = MockLogic.return_value
        mock_logic_instance.logic_heuristic_model.return_value = (
            True  # Simulate a positive heuristic result
        )

        # Create a dummy DataFrame for testing
        data = {
            "subject": ["Test Subject", "re: Reply"],
            "message": ["Test Message Body", "---- forwarded email ----"],
        }
        df = pd.DataFrame(data)

        # Mock dd.from_pandas to return a standard Pandas DataFrame directly
        mock_from_pandas.return_value = df

        # Mock the apply methods for subject and message cleaning
        with patch.object(
            pd.Series,
            "apply",
            side_effect=[
                pd.Series(
                    [["test", "subject"], ["re:", "reply"]]
                ),  # For subject cleaning
                pd.Series(
                    [["test", "message", "body"], ["forwarded", "email"]]
                ),  # For message cleaning
            ],
        ):
            # Mock the final apply call that uses logic_heuristic_model
            with patch.object(
                pd.DataFrame,
                "apply",
                side_effect=lambda func: pd.Series(
                    [
                        mock_logic_instance.logic_heuristic_model(
                            nlp=mock_nlp,
                            subject=row[
                                "subject"
                            ],  # Pass the actual subject from the row
                            message=row[
                                "message"
                            ],  # Pass the actual message from the row
                        )
                        if row["labels"]
                        else False
                        for _, row in df.iterrows()
                    ]
                ),
            ):
                cleaned_df = self.process_data.clean_df(df.copy(), n_partitions=1)

                self.assertIsInstance(cleaned_df, pd.DataFrame)
                self.assertIn("labels", cleaned_df.columns)
                self.assertNotIn("sub_labels", cleaned_df.columns)
                self.assertNotIn("message_labels", cleaned_df.columns)

                # Verify that logic_heuristic_model was called for the first row (True label)
                mock_logic_instance.logic_heuristic_model.assert_called_once()
                mock_logic_instance.logic_heuristic_model.assert_called_with(
                    nlp=mock_nlp,
                    subject=["test", "subject"],
                    message=["test", "message", "body"],
                )


if __name__ == "__main__":
    unittest.main()
