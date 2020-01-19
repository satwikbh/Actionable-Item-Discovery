from re import search

from spacy import load

from Utils.Helper import Helper
from Utils.LoggerUtil import LoggerUtil


class HeuristicRules:
    """
    These rules are adapted from the paper
    "Identifying Business Tasks and Commitments from Email and Chat Conversations"
    TODO: Parsing based on POS tags. Can we use NLTK Regular Expressions for better efficiency?
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = Helper()

    @staticmethod
    def match2(pos, tag1, tag2):
        for tok in pos:
            end = len(tok) - 1
            for index, (i, j) in enumerate(tok, 1):
                if index == end:
                    break
                if j == tag1 and tok[index][1] == tag2:
                    yield ("{} {}".format(i, tok[index][0], tok[index + 1][0]))

    @staticmethod
    def match3(pos, tag1, tag2, tag3):
        yield {""}

    @staticmethod
    def match4(pos, tag1, tag2, tag3, tag4):
        yield {""}

    def rule_1(self, pos):
        """
        A modal verb signals the creation of a commitment.
        For example, the sentence "He will handle the issuance of the LC". contains a modal verb "will"
        It indicates the creation of a commitment.

        (MD) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
        (PRP/PRP$) (MD) (VB/VBD/VBG/VBN/VBP/VBZ)
        :param pos:
        :return:
        """
        match_list = list()
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match3(pos, "MD", "PRP$", "VBZ"))))
        return any(match_list)

    @staticmethod
    def rule_2(sentences):
        """
        A question mark in a sentence indicates a directive commitment creation.
        :return:
        """
        for sentence in sentences:
            if sentence[-1] == "?":
                return True

    def rule_3(self, pos):
        """
        The bi-gram of "please" and an action verb indicates a directive.
        For example, in the sentence "Please review and send along to your attorney as soon as possible",
        the bi-gram please review indicates a directive commitment creation.
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NN", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNS", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNP", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "NNPS", "VBZ"))))

        return any(match_list)

    def rule_4(self, pos):
        """
        The negative verb indicates the presence of a cancelled action.
        For example, in the sentence "Please do not pass the cheque",
        do not is a negative verb but its an actionable sentence none the less.

        (PRP/PRP$) (MD) (RB) (VB/VBD/VBG/VBN/VBP/VBZ)
        (MD) (RB) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBZ"))))

        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VB"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBD"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBG"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBN"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBP"))))
        match_list.append(self.helper.is_list_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBZ"))))

        return any(match_list)

    @staticmethod
    def rule_5(sentences):
        """
        Common abbreviations and acronyms such as ASAP, RSVP, ETA, ETD, ET.
        :return:
        """
        match_list = list()
        for sentence in sentences:
            match_list.append(search('asap', sentence.lower()) is not None)
            match_list.append(search('a.s.a.p', sentence.lower()) is not None)
            match_list.append(search('rsvp', sentence.lower()) is not None)
            match_list.append(search('r.s.v.p', sentence.lower()) is not None)
            match_list.append(search('eta', sentence.lower()) is not None)
            match_list.append(search('e.t.a', sentence.lower()) is not None)
            match_list.append(search('etd', sentence.lower()) is not None)
            match_list.append(search('e.t.d', sentence.lower()) is not None)
            match_list.append(search('et', sentence.lower()) is not None)
            match_list.append(search('et', sentence.lower()) is not None)
        return any(match_list)

    @staticmethod
    def rule_6(sentences):
        """
        Anything which involves completion of a task in the future is an action sentence.
        For examples, in the sentence "It will be posted today and the policy should be drafted by Friday".
        Here today and Friday are the deadlines.
        :return:
        """
        nlp = load("en_core_web_sm")
        match_list = list()
        for sentence in sentences:
            doc = nlp(sentence)
            for X in doc:
                if X.ent_type_ == "TIME":
                    match_list.append(True)
        return any(match_list)

    def rule_7(self, pos):
        """
        A verb followed by pronoun is an action statement
        For example, consider the sentence "Call him"
        :param pos:
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VB", "PRP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBD", "PRP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBG", "PRP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBN", "PRP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBP", "PRP"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBZ", "PRP"))))

        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VB", "PRP$"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBD", "PRP$"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBG", "PRP$"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBN", "PRP$"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBP", "PRP$"))))
        match_list.append(self.helper.is_list_empty(list(self.match2(pos, "VBZ", "PRP$"))))

        return any(match_list)
