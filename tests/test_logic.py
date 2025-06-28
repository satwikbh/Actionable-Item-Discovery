import unittest
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Logic import Logic
from Core.HeuristicRules import HeuristicRules

class TestLogic(unittest.TestCase):

    def setUp(self):
        self.logic = Logic()
        self.mock_rules = Mock(spec=HeuristicRules)
        self.logic.rules = self.mock_rules
        self.mock_nlp = Mock()

    @patch('Core.Logic.pos_tag')
    def test_apply_rules_all_false(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        self.mock_rules.rule_1.return_value = False
        self.mock_rules.rule_2.return_value = False
        self.mock_rules.rule_3.return_value = False
        self.mock_rules.rule_4.return_value = False
        self.mock_rules.rule_5.return_value = False
        self.mock_rules.rule_6.return_value = False
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.configure_mock(__iter__=Mock(return_value=iter([]))) # Make mock_doc iterable
        self.mock_nlp.return_value = mock_doc

        result = self.logic.apply_rules(['word'] * 5, self.mock_nlp)
        self.assertFalse(result)

    @patch('Core.Logic.pos_tag')
    def test_apply_rules_one_true(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        self.mock_rules.rule_1.return_value = False
        self.mock_rules.rule_2.return_value = False
        self.mock_rules.rule_3.return_value = True  # This one is true
        self.mock_rules.rule_4.return_value = False
        self.mock_rules.rule_5.return_value = False
        self.mock_rules.rule_6.return_value = False
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.configure_mock(__iter__=Mock(return_value=iter([]))) # Make mock_doc iterable
        self.mock_nlp.return_value = mock_doc

        result = self.logic.apply_rules(['word'] * 5, self.mock_nlp)
        self.assertTrue(result)

    @patch('Core.Logic.pos_tag')
    def test_apply_rules_time_entity(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        self.mock_rules.rule_1.return_value = False
        self.mock_rules.rule_2.return_value = False
        self.mock_rules.rule_3.return_value = False
        self.mock_rules.rule_4.return_value = False
        self.mock_rules.rule_5.return_value = False
        self.mock_rules.rule_6.return_value = False

        mock_ent = Mock()
        mock_ent.ent_type_ = "TIME"
        mock_doc = Mock()
        mock_doc.ents = [mock_ent]
        mock_doc.configure_mock(__iter__=Mock(return_value=iter([mock_ent]))) # Make mock_doc iterable
        self.mock_nlp.return_value = mock_doc

        result = self.logic.apply_rules(['word'] * 5, self.mock_nlp)
        self.assertTrue(result)

    @patch('Core.Logic.pos_tag')
    def test_logic_heuristic_model_subject_true(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        # Mock apply_rules to return True for subject, False for message
        with patch.object(self.logic, 'apply_rules', side_effect=[True, False]) as mock_apply_rules:
            result = self.logic.logic_heuristic_model(['sub'], ['msg'], self.mock_nlp)
            self.assertTrue(result)
            mock_apply_rules.assert_any_call(['sub'], self.mock_nlp)
            mock_apply_rules.assert_any_call(['msg'], self.mock_nlp)

    @patch('Core.Logic.pos_tag')
    def test_logic_heuristic_model_message_true(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        # Mock apply_rules to return False for subject, True for message
        with patch.object(self.logic, 'apply_rules', side_effect=[False, True]) as mock_apply_rules:
            result = self.logic.logic_heuristic_model(['sub'], ['msg'], self.mock_nlp)
            self.assertTrue(result)
            mock_apply_rules.assert_any_call(['sub'], self.mock_nlp)
            mock_apply_rules.assert_any_call(['msg'], self.mock_nlp)

    @patch('Core.Logic.pos_tag')
    def test_logic_heuristic_model_both_false(self, mock_pos_tag):
        mock_pos_tag.return_value = [('word', 'POS')] * 5
        # Mock apply_rules to return False for both
        with patch.object(self.logic, 'apply_rules', side_effect=[False, False]) as mock_apply_rules:
            result = self.logic.logic_heuristic_model(['sub'], ['msg'], self.mock_nlp)
            self.assertFalse(result)
            mock_apply_rules.assert_any_call(['sub'], self.mock_nlp)
            mock_apply_rules.assert_any_call(['msg'], self.mock_nlp)

if __name__ == '__main__':
    unittest.main()