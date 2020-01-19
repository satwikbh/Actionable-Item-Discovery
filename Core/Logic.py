from nltk import word_tokenize
from nltk.tag import pos_tag

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

    @staticmethod
    def work_on_subject(subject):
        tokens = word_tokenize(subject)
        pos = pos_tag(tokens)

    @staticmethod
    def work_on_message(message):
        tokens = word_tokenize(message)
        pos = pos_tag(tokens)

    def main(self, subject, message):
        """
        Take the subject, message and then validate it against the rules.
        Return true if actionable sentence else false.
        :param subject:
        :param message:
        :return: True or False
        :rtype boolean
        """
        self.work_on_subject(subject)
        self.work_on_message(message)
