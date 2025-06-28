import unittest
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Core.HeuristicRules import HeuristicRules


class TestHeuristicRules(unittest.TestCase):
    def setUp(self):
        self.rules = HeuristicRules()
        # Mock the logger and helper to avoid external dependencies during tests
        self.rules.log = MagicMock()
        self.rules.helper = MagicMock()

    def test_match2(self):
        pos_tags = [("apple", "NN"), ("is", "VBZ"), ("red", "JJ")]
        self.assertEqual(HeuristicRules.match2(pos_tags, "NN", "VBZ"), "apple is")
        self.assertEqual(HeuristicRules.match2(pos_tags, "VBZ", "JJ"), "is red")
        self.assertEqual(HeuristicRules.match2(pos_tags, "NN", "JJ"), "")

    def test_match3(self):
        pos_tags = [("apple", "NN"), ("is", "VBZ"), ("very", "RB"), ("red", "JJ")]
        self.assertEqual(
            HeuristicRules.match3(pos_tags, "NN", "VBZ", "RB"), "apple is very"
        )
        self.assertEqual(
            HeuristicRules.match3(pos_tags, "VBZ", "RB", "JJ"), "is very red"
        )
        self.assertEqual(HeuristicRules.match3(pos_tags, "NN", "RB", "JJ"), "")

    def test_match4(self):
        pos_tags = [
            ("this", "DT"),
            ("apple", "NN"),
            ("is", "VBZ"),
            ("very", "RB"),
            ("red", "JJ"),
        ]
        # Test case where the fourth tag matches
        self.assertEqual(
            HeuristicRules.match4(pos_tags, "DT", "NN", "VBZ", "RB"),
            "this apple is very",
        )
        # Test case where the fourth tag does not match (using 'JJ' instead of 'RB')
        self.assertEqual(HeuristicRules.match4(pos_tags, "DT", "NN", "VBZ", "JJ"), "")
        # Test case with full sequence match
        self.assertEqual(
            HeuristicRules.match4(pos_tags, "NN", "VBZ", "RB", "JJ"),
            "apple is very red",
        )

    def test_rule_1(self):
        # (PRP/PRP$) (MD) (RB) (VB/VBD/VBG/VBN/VBP/VBZ)
        pos_prp_md_rb_vb = [("I", "PRP"), ("will", "MD"), ("not", "RB"), ("go", "VB")]
        # (MD) (RB) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
        pos_md_rb_prp_vb = [("will", "MD"), ("not", "RB"), ("I", "PRP"), ("go", "VB")]

        self.rules.helper.is_list_not_empty.side_effect = lambda x: bool(x)

        # Mock match4 to return a match for the specific pattern being tested, and empty for others
        # This ensures that the mock isn't exhausted by multiple calls within rule_1
        def mock_match4_rule1_prp(pos, t1, t2, t3, t4):
            if t1 == "PRP" and t2 == "MD" and t3 == "RB" and t4 == "VB":
                return ["I will not go"]
            return []

        def mock_match4_rule1_md(pos, t1, t2, t3, t4):
            if t1 == "MD" and t2 == "RB" and t3 == "PRP" and t4 == "VB":
                return ["will not I go"]
            return []

        with unittest.mock.patch.object(
            HeuristicRules, "match4", side_effect=mock_match4_rule1_prp
        ):
            self.assertTrue(self.rules.rule_1(pos_prp_md_rb_vb))

        with unittest.mock.patch.object(
            HeuristicRules, "match4", side_effect=mock_match4_rule1_md
        ):
            self.assertTrue(self.rules.rule_1(pos_md_rb_prp_vb))

        with unittest.mock.patch.object(HeuristicRules, "match4", return_value=[]):  # No match
            pos_no_match = [("I", "PRP"), ("am", "VBP"), ("going", "VBG")]
            self.assertFalse(self.rules.rule_1(pos_no_match))

    def test_rule_2(self):
        pos_md_prp_vb = [("will", "MD"), ("I", "PRP"), ("go", "VB")]
        self.rules.helper.is_list_not_empty.side_effect = lambda x: bool(x)

        def mock_match3_rule2(pos, t1, t2, t3):
            if t1 == "MD" and t2 == "PRP" and t3 == "VB":
                return ["will I go"]
            return []

        with unittest.mock.patch.object(
            HeuristicRules, "match3", side_effect=mock_match3_rule2
        ):
            self.assertTrue(self.rules.rule_2(pos_md_prp_vb))

        with unittest.mock.patch.object(
            HeuristicRules, "match3", return_value=[]
        ):  # No match
            pos_no_match = [("I", "PRP"), ("am", "VBP"), ("going", "VBG")]
            self.assertFalse(self.rules.rule_2(pos_no_match))

    def test_rule_3(self):
        pos_nn_vb = [("please", "NN"), ("go", "VB")]
        tokens_with_please = ["please", "go"]
        tokens_without_please_no_match = ["you", "go"]
        pos_no_match_no_please = [("you", "PRP"), ("go", "VB")]

        self.rules.helper.is_list_not_empty.side_effect = lambda x: bool(x)

        def mock_match2_rule3(pos, t1, t2):
            if t1 == "NN" and t2 == "VB":
                return ["please go"]
            return []

        with unittest.mock.patch.object(
            HeuristicRules, "match2", side_effect=mock_match2_rule3
        ):
            self.assertTrue(self.rules.rule_3(pos_nn_vb, tokens_with_please))

        # Test with "please" in tokens but no POS match
        with unittest.mock.patch.object(HeuristicRules, "match2", return_value=[]):
            self.assertTrue(
                self.rules.rule_3(pos_no_match_no_please, tokens_with_please)
            )

        # Test with no "please" and no POS match
        with unittest.mock.patch.object(HeuristicRules, "match2", return_value=[]):
            self.assertFalse(
                self.rules.rule_3(
                    pos_no_match_no_please, tokens_without_please_no_match
                )
            )

    def test_rule_4(self):
        pos_vb_prp = [("call", "VB"), ("him", "PRP")]
        self.rules.helper.is_list_not_empty.side_effect = lambda x: bool(x)

        def mock_match2_rule4(pos, t1, t2):
            if t1 == "VB" and t2 == "PRP":
                return ["call him"]
            return []

        with unittest.mock.patch.object(
            HeuristicRules, "match2", side_effect=mock_match2_rule4
        ):
            self.assertTrue(self.rules.rule_4(pos_vb_prp))

        with unittest.mock.patch.object(
            HeuristicRules, "match2", return_value=[]
        ):  # No match
            pos_no_match = [("he", "PRP"), ("calls", "VBZ")]
            self.assertFalse(self.rules.rule_4(pos_no_match))

    def test_rule_5(self):
        tokens_with_q = ["what", "is", "this", "?"]
        tokens_without_q = ["this", "is", "it"]
        self.assertTrue(HeuristicRules.rule_5(tokens_with_q))
        self.assertFalse(HeuristicRules.rule_5(tokens_without_q))

    def test_rule_6(self):
        tokens_with_asap = ["send", "it", "asap"]
        tokens_with_rsvp = ["please", "rsvp", "by", "friday"]
        tokens_with_eta = ["what", "is", "the", "eta"]
        tokens_with_etd = ["when", "is", "the", "etd"]
        tokens_with_et = ["let's", "et", "tomorrow"]
        tokens_no_abbr = ["send", "it", "now"]

        self.assertTrue(HeuristicRules.rule_6(tokens_with_asap))
        self.assertTrue(HeuristicRules.rule_6(tokens_with_rsvp))
        self.assertTrue(HeuristicRules.rule_6(tokens_with_eta))
        self.assertTrue(HeuristicRules.rule_6(tokens_with_etd))
        self.assertTrue(HeuristicRules.rule_6(tokens_with_et))
        self.assertFalse(HeuristicRules.rule_6(tokens_no_abbr))

        # Test with dots
        tokens_with_asap_dots = ["send", "it", "a.s.a.p"]
        self.assertTrue(HeuristicRules.rule_6(tokens_with_asap_dots))

    def test_rule_7(self):
        mock_nlp = MagicMock()

        # Mock the behavior of nlp("...").ents
        # Case 1: Contains a TIME entity
        doc_with_time = MagicMock()
        token_time = MagicMock()
        token_time.ent_type_ = "TIME"
        token_other = MagicMock()
        token_other.ent_type_ = "PERSON"
        doc_with_time.__iter__.return_value = [
            token_time,
            token_other,
        ]  # Make the doc iterable

        # Case 2: Does not contain a TIME entity
        doc_without_time = MagicMock()
        token_person = MagicMock()
        token_person.ent_type_ = "PERSON"
        doc_without_time.__iter__.return_value = [token_person]  # Make the doc iterable

        tokens_future_related = ["meeting", "is", "tomorrow"]
        tokens_not_future_related = ["meeting", "was", "yesterday"]

        # Scenario 1: "tomorrow" is recognized as TIME
        mock_nlp.return_value = doc_with_time
        self.assertTrue(HeuristicRules.rule_7(mock_nlp, tokens_future_related))

        # Scenario 2: No TIME entity found
        mock_nlp.return_value = doc_without_time
        self.assertFalse(HeuristicRules.rule_7(mock_nlp, tokens_not_future_related))


if __name__ == "__main__":
    unittest.main()
