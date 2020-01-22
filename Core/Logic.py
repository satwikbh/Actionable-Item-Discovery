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

    def apply_rules(self, text_tokens, nlp):
        pos = pos_tag(text_tokens)
        match_list = list()

        if len(pos) >= 4:
            match_list.append(self.rules.rule_1(pos))
        if len(pos) >= 3:
            match_list.append(self.rules.rule_2(pos))
        if len(pos) >= 2:
            match_list.append(self.rules.rule_3(pos, text_tokens))
            match_list.append(self.rules.rule_4(pos))

        match_list.append(self.rules.rule_5(text_tokens))
        match_list.append(self.rules.rule_6(text_tokens))
        # Spacy has known issues when parallelizing hence, it wont work for such
        doc = nlp(" ".join(text_tokens))
        match_list.append(any([True if X.ent_type_ == "TIME" else False for X in doc]))
        return any(match_list)

    def logic_heuristic_model(self, subject, message, nlp):
        """
        Take the subject, message and then validate it against the rules.
        Return true if actionable sentence else false.
        """
        sub_val = self.apply_rules(subject, nlp)
        msg_val = self.apply_rules(message, nlp)

        return sub_val or msg_val
