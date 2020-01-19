from Core.Logic import Logic
from Utils.LoggerUtil import LoggerUtil


class ProcessData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.logic = Logic()

    @staticmethod
    def get_regex_pattern():
        pattern = r'(?:Message-ID: )([\s\S]*)(?:\n)'
        pattern += r'(?:Date: )([\s\S]*)(?:\n)'
        pattern += r'(?:From: )(.*)'
        pattern += r'(?:(?:(?:\n)(?=(?:To:) )(?:To: )([\s\S]*)(?:\n))|(?:(?:\n)(?!(?:To: ))))'
        pattern += r'(?:Subject: )([\s\S]*)(?:\n)'
        pattern += r'(?:Mime-Version: )([\s\S]*)(?:\n)'
        pattern += r'(?:Content-Type: )([\s\S]*)(?:\n)'
        pattern += r'(?:Content-Transfer-Encoding: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-From: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-To: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-cc: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-bcc: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-Folder: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-Origin: )([\s\S]*)(?:\n)'
        pattern += r'(?:X-FileName: )(.*)(?:\n)'
        pattern += r'([\s\S]*$)'
        return pattern

    def get_sub_message(self, df):
        regex_pattern = self.get_regex_pattern()
        new_df = df.message.str.extract(regex_pattern)[[4, 15]]
        new_df.columns = ["subject", "message"]
        # new_df.insert(0, 'file', df['file'])
        return new_df

    @staticmethod
    def clean_text(subject, message):
        """
        Don't consider Reply mails (if re: in subject)
        Don't consider forward mails (if ---forward in message)
        :param subject:
        :param message:
        :return:
        """
        if "re:" in subject.lower() or "---- forwarded" in message.lower():
            return [""] * 2
        else:
            return subject.strip(), message.strip()

    def clean_df(self, df):
        labels = list()
        for index, row in df.iterrows():
            subject, message = row['subject'], row['message']
            subject, message = self.clean_text(subject, message)
            if subject != "" and message != "":
                label = self.logic.main(subject, message)
                labels.append(label)
            else:
                labels.append(False)
        return labels
