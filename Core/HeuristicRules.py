from Utils.LoggerUtil import LoggerUtil


class HeuristicRules:
    """
    These rules are adapted from the paper
    "Identifying Business Tasks and Commitments from Email and Chat Conversations"
    TODO: Write Regular Expressions for the below rules.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def modal_verb_rule():
        """
        A modal verb signals the creation of a commitment.
        For example, the sentence "He will handle the issuance of the LC". contains a modal verb "will"
        It indicates the creation of a commitment.

        (MD) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)
        (PRP/PRP$) (MD) (VB/VBD/VBG/VBN/VBP/VBZ)

        :return:
        """
        pass

    @staticmethod
    def rule_2():
        """
        A question mark in a sentence indicates a directive commitment creation.
        :return:
        """
        pass

    @staticmethod
    def rule_3():
        """
        The bi-gram of "please" and an action verb indicates a directive.
        For example, in the sentence "Please review and send along to your attorney as soon as possible",
        the bi-gram please review indicates a directive commitment creation.
        :return:
        """
        pass

    @staticmethod
    def rule_4():
        """
        The negative verb indicates the presence of a cancelled action.
        For example, in the sentence "Please do not pass the cheque",
        do not is a negative verb but its an actionable sentence none the less.

        (PRP/PRP$) (MD) (RB) (VB/VBD/VBG/VBN/VBP/VBZ)
        (MD) (RB) (PRP/PRP$) (VB/VBD/VBG/VBN/VBP/VBZ)

        :return:
        """
        pass

    @staticmethod
    def rule_5():
        """
        Presence of words such as Call or Text or Ping followed by a PRP.
        Common abbreviations and acronyms such as ASAP, RSVP, ETA, ETD, ET.
        :return:
        """
        pass

    @staticmethod
    def rule_6():
        """
        Anything which involves completion of a task in the future is an action sentence.
        For examples, in the sentence "It will be posted today and the policy should be drafted by Friday".
        Here today and Friday are the deadlines.
        :return:
        """
        pass
