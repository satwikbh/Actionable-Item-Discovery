from nltk import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from numpy import any

from Core.HeuristicRules import HeuristicRules
from Utils.LoggerUtil import LoggerUtil


class Logic:
    """
    TODO:
    1. Match with regex
    2. If the sentence obeys the heuristic rules, classify it as actionable.
    3. Else as non-actionable.

    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.rules = HeuristicRules()

    def apply_rules(self, pos, sentences):
        match_list = list()
        match_list.append(self.rules.rule_1(pos))
        match_list.append(self.rules.rule_2(sentences))
        match_list.append(self.rules.rule_3(pos))
        match_list.append(self.rules.rule_4(pos))
        match_list.append(self.rules.rule_5(sentences))
        match_list.append(self.rules.rule_6(sentences))
        match_list.append(self.rules.rule_7(pos))
        return any(match_list)

    def work_on_text(self, text):
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)
        pos = pos_tag(tokens)
        return self.apply_rules(pos, sentences)

    def main(self, subject, message):
        """
        Take the subject, message and then validate it against the rules.
        Return true if actionable sentence else false.
        :param subject:
        :param message:
        :return: True or False
        :rtype boolean
        """
        sub_result = self.work_on_text(text=subject)
        msg_result = self.work_on_text(text=message)
        if sub_result or msg_result:
            return True
        else:
            return False
