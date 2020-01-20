from collections import Counter

from pandas import read_csv, concat

from Core.ProcessData import ProcessData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class ReadData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.process_data = ProcessData()

    def main(self, n_rows):
        data_path = self.config["data_path"]
        chunksize = self.config["chunksize"]
        df_list, labels = list(), list()
        for i in range(int(n_rows / chunksize)):
            self.log.info("Iter : {}".format(i))
            chunk = read_csv(data_path + "/" + "emails.csv", chunksize=chunksize)
            df = next(chunk)
            df = self.process_data.get_sub_message(df)
            df_list.append(df)
            labels += self.process_data.clean_df(df)
            self.log.info("Label Stats : {}".format(Counter(labels)))
        return concat(df_list), labels


if __name__ == '__main__':
    read_data = ReadData()
    read_data.main(n_rows=5000)
