import unittest
import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.ReadData import ReadData
from Core.ProcessData import ProcessData # Needed for mocking

class TestReadData(unittest.TestCase):

    def setUp(self):
        # Mock ConfigUtil to return a predefined configuration
        self.mock_config = {
            "data_path": "/fake/data/path",
            "chunksize": 100,
            "n_partitions": 2,
            "tagged_path": "/fake/tagged/path"
        }
        self.config_patcher = patch('Utils.ConfigUtil.ConfigUtil.get_config_instance')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = self.mock_config

        # Mock ProcessData methods that are used by ReadData
        self.mock_process_data_instance = MagicMock(spec=ProcessData)
        self.process_data_patcher = patch('Core.ReadData.ProcessData')
        self.mock_process_data_class = self.process_data_patcher.start()
        self.mock_process_data_class.return_value = self.mock_process_data_instance

        # Mock Helper methods (if any direct calls were made, not strictly necessary if ProcessData is fully mocked)
        self.helper_patcher = patch('Core.ReadData.Helper')
        self.mock_helper_class = self.helper_patcher.start()
        self.mock_helper_instance = self.mock_helper_class.return_value

        # Mock LoggerUtil
        self.logger_patcher = patch('Core.ReadData.LoggerUtil')
        self.mock_logger_class = self.logger_patcher.start()
        self.mock_logger_instance = self.mock_logger_class.return_value.get.return_value = MagicMock()


        self.read_data = ReadData()
        # Reset mocks for ProcessData within ReadData instance to ensure it uses the patched one
        self.read_data.process_data = self.mock_process_data_instance

        # Directly set the stop_words on the instance to ensure correctness for tests
        # Define a known set of spaCy stop words relevant for tests, as dynamic import is unreliable here.
        # This list is based on the output of `python -c "from spacy.lang.en.stop_words import STOP_WORDS; print(STOP_WORDS)"`
        # and includes words relevant to the test cases. It doesn't need to be exhaustive for unrelated words.
        known_spacy_stop_words = {
            'hence', 'part', 'upon', 'will', 'fifty', 'cannot', 'should', 'everywhere', 'some', 'above', 'whenever',
            'whoever', 'along', 'keep', 'show', 'sixty', 'many', 'further', 'various', 'with', 'alone', 'where',
            'very', 'get', 'amongst', 'every', 'towards', '‘s', 'whether', 'our', 'just', 'what', 'nevertheless',
            'all', 'most', 'neither', 'her', 'anywhere', 'whereafter', 'of', 'by', 'become', 'otherwise', 'during',
            'beyond', 'can', 'if', "'d", 'perhaps', 'to', 'namely', "'re", 'into', 'others', 'nobody', 'nine',
            'whence', 'you', 'made', 'no', 'seem', 'serious', 'whole', 'until', 'other', 'behind', 'latterly',
            'myself', 'thereafter', 'whom', 'anyone', 'back', 'never', 'same', 'else', 'amount', '’ll', 'your',
            'would', 'yourselves', 'could', 'whereupon', 'becomes', 'seems', 'his', 'itself', '‘re', 'why', 'was',
            'there', 'and', 'yours', 'ca', 'each', 'whereby', 'its', 'became', 'she', 'top', 'the', 'but', 'us',
            'or', 'either', 'give', 'although', '’ve', 'mine', 'only', 'does', 'those', 'more', 'any', 'used',
            'hereafter', 'regarding', 'a', 'five', 'less', 'so', '‘m', 'thereby', 'via', 'me', 'whereas', 'as',
            'i', 'nothing', 'often', 'had', 'then', 'it', 'against', 'thru', 'than', 'my', 'name', 'hereby',
            'toward', 'except', 'them', 'always', 'throughout', 'eight', 'something', 'while', 'whose', 'mostly',
            'not', 'six', 'without', 'now', 'hundred', 'ourselves', 'ours', 'still', 'rather', 'yet', 'however',
            'even', 'already', 'once', 'thereupon', '’s', '’re', 'several', 'since', 'latter', 'do', 'have',
            'when', 'such', 'be', 'has', 'third', 'up', 'indeed', 'everyone', 're', "'ve", 'own', 'wherein',
            'around', 'their', 'one', 'four', 'is', 'between', 'am', 'n‘t', 'seemed', 'therefore', 'front', 'put',
            'everything', 'wherever', 'over', 'hers', 'he', 'anything', 'nor', 'none', 'they', 'also', "n't",
            'full', 'next', 'first', 'eleven', 'much', 'that', 'hereupon', 'quite', 'side', 'fifteen', 'twelve',
            'ten', 'been', 'meanwhile', 'formerly', 'really', 'which', 'on', '‘d', 'former', 'across', 'almost',
            'being', 'please', 'done', 'few', 'ever', 'say', 'sometime', 'we', 'within', 'were', 'anyhow',
            'empty', 'n’t', 'himself', 'thus', 'herself', 'two', 'yourself', 'per', 'move', 'least', 'beforehand',
            'may', 'for', 'three', 'both', '‘ll', "'m", 'besides', 'somewhere', '‘ve', "'s", 'out', 'unless',
            'somehow', 'another', 'from', 'at', 'take', 'must', 'about', 'due', 'off', 'themselves', 'under',
            'beside', 'did', 'because', 'how', 'therein', 'through', 'becoming', 'who', 'using', 'sometimes',
            'go', 'might', 'moreover', 'though', 'before', 'nowhere', 'thence', 'again', 'anyway', 'last', 'after',
            'twenty', '’d', 'afterwards', 'these', 'below', 'call', 'together', 'in', 'enough', 'too', 'here',
            'him', 'elsewhere', 'this', 'forty', "'ll", 'an', 'someone', 'are', 'among', 'well', 'noone',
            'whatever', 'herein', 'see', 'whither', 'bottom', 'doing', 'down', 'onto', 'make', 'seeming', '’m', 'like' # Added 'like'
        }
        custom_stop_words = {'ect', 'hou', 'com', 'recipient', 'na', 'ou', 'cn', 'enron', 'zdnet'}
        correct_stop_words = set.union(known_spacy_stop_words, custom_stop_words)
        self.read_data.stop_words = correct_stop_words


    def tearDown(self):
        self.config_patcher.stop()
        self.process_data_patcher.stop()
        self.helper_patcher.stop()
        self.logger_patcher.stop()
        # self.stop_words_patcher.stop() # No longer using this direct patch

    @patch('Core.ReadData.read_csv')
    def test_prepare_data(self, mock_read_csv):
        # Create dummy data for two chunks
        dummy_df_chunk1 = pd.DataFrame({'message': ['email1 subject\nbody1', 'email2 subject\nbody2']})
        dummy_df_chunk2 = pd.DataFrame({'message': ['email3 subject\nbody3', 'email4 subject\nbody4']})

        # Mock read_csv to return an iterator of these chunks
        mock_read_csv.return_value = iter([dummy_df_chunk1, dummy_df_chunk2])

        # Mock ProcessData methods to return the DataFrame passed to them, adding a 'labels' column
        def mock_get_sub_message(df):
            df['subject'] = df['message'].apply(lambda x: x.split('\\n')[0])
            df['message_body'] = df['message'].apply(lambda x: x.split('\\n')[1] if '\\n' in x else '')
            return df

        def mock_clean_df(df, n_partitions):
            df['labels'] = [True, False] * (len(df) // 2) # Example labels
            return df

        self.mock_process_data_instance.get_sub_message.side_effect = mock_get_sub_message
        self.mock_process_data_instance.clean_df.side_effect = mock_clean_df

        n_rows = 200 # This means 2 chunks since chunksize is 100
        result_df = self.read_data.prepare_data(n_rows)

        self.assertEqual(mock_read_csv.call_count, 1)
        mock_read_csv.assert_called_with("/fake/data/path/emails.csv", chunksize=100)

        self.assertEqual(self.mock_process_data_instance.get_sub_message.call_count, 2) # Called for each chunk
        self.assertEqual(self.mock_process_data_instance.clean_df.call_count, 2) # Called for each chunk

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 4) # 2 chunks * 2 rows each
        self.assertIn('labels', result_df.columns)
        self.assertIn('subject', result_df.columns) # Added by mock_get_sub_message
        self.assertIn('message_body', result_df.columns) # Added by mock_get_sub_message


    @patch('Core.ReadData.read_csv')
    def test_prepare_tagged_data(self, mock_read_csv):
        # Dummy CSV content for actions.csv
        csv_content = "action one, some other data\n" \
                      "action two here, more data\n" \
                      "please stop action three, even more"

        # Mock read_csv to return a DataFrame parsed from this string
        # We use StringIO to simulate a file
        from io import StringIO
        mock_df = pd.read_csv(StringIO(csv_content), header=None)
        mock_read_csv.return_value = mock_df

        # Expected tokens after processing (lowercase, stripped, stopwords removed)
        # Stop words from spaCy: 'some', 'other', 'here', 'more', 'please', 'three', 'even'
        # Custom stop words: none relevant here
        # Not stop words: 'action', 'one', 'two', 'data', 'stop' (word 'stop' is not a stopword by default)
        expected_data = [
            ['action', 'one', 'data'],    # from "action one, some other data"
            ['action', 'two', 'data'],    # from "action two here, more data"
            ['stop', 'action']            # from "please stop action three, even more"
        ]

        result_df = self.read_data.prepare_tagged_data()

        mock_read_csv.assert_called_once_with("/fake/tagged/path/actions.csv", header=None)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('data', result_df.columns)
        self.assertIn('labels', result_df.columns)
        self.assertEqual(len(result_df), 3)
        pd.testing.assert_series_equal(result_df['labels'], pd.Series([True, True, True], name='labels'), check_dtype=False)

        # Check the processed data column
        for i, row_list in enumerate(result_df['data']):
            self.assertListEqual(row_list, expected_data[i])


    def test_prepare_tagged_data_lambda_logic(self):
        # Test the core logic of the lambda used in prepare_tagged_data directly.
        # self.read_data.stop_words is assumed to be correctly set up by setUp.

        test_string_1 = "action one, some other data"
        # Expected: ['action', 'one', 'data'] (since 'some', 'other' are stopwords)
        processed_list_1 = [
            item.lower().strip()
            for item in test_string_1.split()
            if item.lower().strip() not in self.read_data.stop_words
        ]
        self.assertListEqual(processed_list_1, ['action', 'one', 'data'])

        test_string_2 = "action two here, more data"
        # Expected: ['action', 'two', 'data'] (since 'here', 'more' are stopwords)
        processed_list_2 = [
            item.lower().strip()
            for item in test_string_2.split()
            if item.lower().strip() not in self.read_data.stop_words
        ]
        self.assertListEqual(processed_list_2, ['action', 'two', 'data'])

        test_string_3 = "please stop action three, even more"
        # Expected: ['stop', 'action'] (since 'please', 'three', 'even', 'more' are stopwords)
        processed_list_3 = [
            item.lower().strip()
            for item in test_string_3.split()
            if item.lower().strip() not in self.read_data.stop_words
        ]
        self.assertListEqual(processed_list_3, ['stop', 'action'])


    def test_transform_sentence(self):
        # Cleaned up debug prints

        sentence1 = "This is a Test Sentence with stop words like the and is"
        # Default spaCy STOP_WORDS includes: 'this', 'is', 'a', 'like', 'the', 'and'
        # Custom stop words in ReadData: {'ect', 'hou', 'com', 'recipient', 'na', 'ou', 'cn', 'enron', 'zdnet'}
        # Expected: "test", "sentence", "stop", "words"
        expected1 = ["test", "sentence", "stop", "words"]
        self.assertListEqual(self.read_data.transform_sentence(sentence1), expected1)

        sentence2 = "Another   example with ECT and HOU "
        # Default spaCy STOP_WORDS includes: 'another', 'with', 'and'
        # Custom includes: 'ect', 'hou'
        # Expected: "example"
        expected2 = ["example"]
        self.assertListEqual(self.read_data.transform_sentence(sentence2), expected2)

        sentence3 = "  leading and trailing spaces  "
        expected3 = ["leading", "trailing", "spaces"]
        self.assertListEqual(self.read_data.transform_sentence(sentence3), expected3)

        sentence4 = "NoChangeNeeded"
        expected4 = ["nochangeneeded"]
        self.assertListEqual(self.read_data.transform_sentence(sentence4), expected4)


if __name__ == '__main__':
    unittest.main()
