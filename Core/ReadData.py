from collections import Counter

from nltk.stem import PorterStemmer
from numpy import ones
from pandas import read_csv, concat, DataFrame
from spacy.lang.en.stop_words import STOP_WORDS

from Core.ProcessData import ProcessData
from Utils.ConfigUtil import ConfigUtil
from Utils.Helper import Helper
from Utils.LoggerUtil import LoggerUtil


class ReadData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.process_data = ProcessData()
        self.helper = Helper()
        self.ps = PorterStemmer()
        self.stop_words = set.union(STOP_WORDS, {'ect', 'hou', 'com', 'recipient', 'na', 'ou', 'cn', 'enron', 'zdnet'})

    def prepare_data(self, n_rows):
        data_path = self.config["data_path"]
        chunksize = self.config["chunksize"]
        n_partitions = self.config["n_partitions"]
        df_list = list()
        chunk = read_csv(data_path + "/" + "emails.csv", chunksize=chunksize)
        for i in range(int(n_rows / chunksize)):
            self.log.info("Iter : {}".format(i))
            df = next(chunk)
            df = self.process_data.get_sub_message(df)
            self.process_data.clean_df(df, n_partitions)
            df_list.append(df)
            self.log.info("Label Stats : {}".format(Counter(df["labels"])))
        return concat(df_list)

    def prepare_tagged_data(self):
        tagged_path = self.config["tagged_path"]
        tagged_data = read_csv(tagged_path + "/" + "actions.csv", header=None)
        rows = len(tagged_data)
        tagged_data[0] = tagged_data[0].apply(
            lambda x: ["".join(item.lower().strip()) for item in x.split() if x.lower().strip() not in self.stop_words])
        df = DataFrame.from_dict({"data": tagged_data[0].values, "labels": ones(rows, dtype=bool)})
        return df

    def transform_sentence(self, sentence):
        return [item.lower().strip() for item in sentence.split() if item.lower().strip() not in self.stop_words]


if __name__ == '__main__':
    read_data = ReadData()
    read_data.prepare_data(n_rows=5000)
