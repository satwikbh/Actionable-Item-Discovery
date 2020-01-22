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
        for i, j in zip(pos, pos[1:]):
            if i[1] == tag1 and j[1] == tag2:
                return " ".join([i[0], j[0]])
        return ""

    @staticmethod
    def match3(pos, tag1, tag2, tag3):
        for x, y, z in zip(pos, pos[1:], pos[2:]):
            if x[1] == tag1 and y[1] == tag2 and z[1] == tag3:
                return " ".join([x[0], y[0], z[0]])
        return ""

    @staticmethod
    def match4(pos, tag1, tag2, tag3, tag4):
        for p, q, r, s in zip(pos, pos[1:], pos[2:], pos[3:]):
            if p[1] == tag1 and q[1] == tag2 and r[1] == tag3 and s == tag4:
                return " ".join([p[0], q[0], r[0], s[0]])
        return ""

    def rule_1(self, pos):
        """
        The negative verb indicates the presence of a cancelled action.
        For example, in the sentence "Please do not pass the cheque",
        do not is a negative verb but its an actionable sentence none the less.

        (PRP/PRP$) (MD) (RB) (VB/VBD/VBG/VBN/VBP/VBZ)
        (MD) (RB) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP", "MD", "RB", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "PRP$", "MD", "RB", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match4(pos, "MD", "RB", "PRP$", "VBZ"))))

        return any(match_list)

    def rule_2(self, pos):
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
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match3(pos, "MD", "PRP$", "VBZ"))))
        return any(match_list)

    def rule_3(self, pos, text_tokens):
        """
        The bi-gram of "please" and an action verb indicates a directive.
        For example, in the sentence "Please review and send along to your attorney as soon as possible",
        the bi-gram please review indicates a directive commitment creation.
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NN", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNS", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNP", "VBZ"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VB"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VBD"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VBG"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VBN"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VBP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "NNPS", "VBZ"))))

        if 'please' in text_tokens:
            match_list.append(True)

        return any(match_list)

    def rule_4(self, pos):
        """
        A verb followed by pronoun is an action statement
        For example, consider the sentence "Call him"
        :param pos:
        :return:
        """
        match_list = list()

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VB", "PRP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBD", "PRP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBG", "PRP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBN", "PRP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBP", "PRP"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBZ", "PRP"))))

        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VB", "PRP$"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBD", "PRP$"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBG", "PRP$"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBN", "PRP$"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBP", "PRP$"))))
        match_list.append(self.helper.is_list_not_empty(list(self.match2(pos, "VBZ", "PRP$"))))

        return any(match_list)

    @staticmethod
    def rule_5(text_tokens):
        """
        A question mark in a sentence indicates a directive commitment creation.
        :return:
        """
        return "?" in text_tokens

    @staticmethod
    def rule_6(text_tokens):
        """
        Common abbreviations and acronyms such as ASAP, RSVP, ETA, ETD, ET.
        :return:
        """
        match_list = list()

        match_list.append(['asap' in text_tokens])
        match_list.append(['a.s.a.p' in text_tokens])
        match_list.append(['rsvp' in text_tokens])
        match_list.append(['r.s.v.p' in text_tokens])
        match_list.append(['eta' in text_tokens])
        match_list.append(['e.t.a' in text_tokens])
        match_list.append(['etd' in text_tokens])
        match_list.append(['e.t.d' in text_tokens])
        match_list.append(['et' in text_tokens])
        match_list.append(['e.t' in text_tokens])

        return any(match_list)

    @staticmethod
    def rule_7(nlp, text_tokens):
        """
        Anything which involves completion of a task in the future is an action sentence.
        For examples, in the sentence "It will be posted today and the policy should be drafted by Friday".
        Here today and Friday are the deadlines.
        :return:
        """
        match_list = list()
        doc = nlp(" ".join(text_tokens))
        for X in doc:
            if X.ent_type_ == "TIME":
                match_list.append(True)
        return any(match_list)
