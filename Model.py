from sklearn.model_selection import train_test_split

from Core.ReadData import ReadData
from Utils.LoggerUtil import LoggerUtil


class Model:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.read_data = ReadData()

    @staticmethod
    def split_data(df, labels):
        x_train, x_test, y_train, y_test = train_test_split(df, labels, stratify=labels)
        return x_train, x_test, y_train, y_test

    def main(self):
        df, labels = self.read_data.main()
        x_train, x_test, y_train, y_test = self.split_data(df, labels)


if __name__ == '__main__':
    model = Model()
    model.main()
