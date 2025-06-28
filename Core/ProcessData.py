from dask import dataframe as dd
from nltk.stem import PorterStemmer
from spacy import load
from spacy.lang.en.stop_words import STOP_WORDS

from Core.Logic import Logic
from Utils.LoggerUtil import LoggerUtil


class ProcessData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.logic = Logic()
        self.ps = PorterStemmer()

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
        if "re:" in subject or "---- forwarded" in message:
            return ["", ""]
        return [subject, message]

    def clean_df(self, df, n_partitions):
        stop_words = set.union(STOP_WORDS, {'ect', 'hou', 'com', 'recipient', 'na', 'ou', 'cn', 'enron', 'zdnet'})
        nlp = load("en_core_web_sm")

        df["subject"] = dd.from_pandas(df["subject"], npartitions=n_partitions).map_partitions(
            lambda my_df: my_df.apply(
                lambda x: [item.lower().strip() for item in x.split() if
                           item.lower().strip() not in stop_words]
            )
        ).compute()

        df["message"] = dd.from_pandas(df["message"], npartitions=n_partitions).map_partitions(
            lambda my_df: my_df.apply(
                lambda x: [item.lower().strip() for item in x.split() if item.lower().strip() not in stop_words]
            )
        ).compute()

        df["sub_labels"] = df["subject"].apply(
            lambda x: False if "re:" in x else True
        )
        df["message_labels"] = df["message"].apply(
            lambda x: False if "----------------------" in x or "forwarded" in x else True
        )

        df["labels"] = df["sub_labels"] & df["message_labels"]
        df.drop("sub_labels", inplace=True, axis=1)
        df.drop("message_labels", inplace=True, axis=1)

        df["labels"] = dd.from_pandas(df, npartitions=n_partitions).map_partitions(
            lambda my_df: my_df.T.apply(
                lambda x: self.logic.logic_heuristic_model(nlp=nlp, subject=x["subject"], message=x["message"]) if x[
                    "labels"] else False
            )
        ).compute()
        return df
