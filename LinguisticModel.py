from numpy import zeros, concatenate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Core.ReadData import ReadData
from Models.Adaboost import Adaboost
from Models.DecisionTree import DecisionTree
from Models.ExtraTree import ExtraTree
from Models.LRModel import LRModel
from Models.MLP import MLP
from Models.NaiveBayes import NaiveBayes
from Models.RandomForest import RandomForest
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class LinguisticModel:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.read_data = ReadData()
        self.bayes = NaiveBayes()
        self.lr = LRModel()
        self.dt = DecisionTree()
        self.rf = RandomForest()
        self.et = ExtraTree()
        self.adaboost = Adaboost()
        self.mlp = MLP()

    @staticmethod
    def split_data(df, labels):
        x_train, x_test, y_train, y_test = train_test_split(df, labels, stratify=labels, shuffle=True)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def custom_word_tokenizer(tokens):
        return tokens

    def vectorize_data(self, x_train, x_test):
        """
        While we needed subject to estimate if the mail contains action item or not,
        we don't need it for model selection.
        :return:
        """
        vectorizer = TfidfVectorizer(
            tokenizer=self.custom_word_tokenizer,
            sublinear_tf=True,
            analyzer='word',
            lowercase=False,
            max_features=1000
        )

        x_train_features = vectorizer.fit_transform(x_train)
        x_test_features = vectorizer.fit_transform(x_test)
        return x_train_features, x_test_features

    @staticmethod
    def get_tagged_dataset(df):
        return df[~df["labels"]]["message"].head(n=1250), zeros(1250, dtype=bool)

    def train_models(self, x_train_features, x_test_features, y_train, y_test):
        self.bayes.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.lr.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.dt.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.rf.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.et.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.adaboost.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)
        self.mlp.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test)

    def main(self):
        tagged_data_df = self.read_data.prepare_tagged_data()

        untagged_data_df = self.read_data.prepare_data(n_rows=2500).sample(frac=1)
        untagged_data, untagged_labels = self.get_tagged_dataset(untagged_data_df)

        data = concatenate((tagged_data_df["data"], untagged_data))
        labels = concatenate((tagged_data_df["labels"], untagged_labels))

        x_train, x_test, y_train, y_test = self.split_data(data, labels)
        x_train_features, x_test_features = self.vectorize_data(x_train, x_test)

        self.train_models(x_train_features, x_test_features, y_train, y_test)


if __name__ == '__main__':
    model = LinguisticModel()
    model.main()
