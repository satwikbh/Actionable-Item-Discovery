from pandas import read_csv

from Core.ProcessData import ProcessData
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class ReadData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.process_data = ProcessData()

    def main(self):
        data_path = self.config["data_path"]
        df = read_csv(data_path + "/" + "emails.csv", nrows=100)
        df = self.process_data.get_sub_message(df)
        labels = self.process_data.clean_df(df)
        return df, labels


if __name__ == '__main__':
    read_data = ReadData()
    read_data.main()
