from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Core.ReadData import ReadData
from Models.Adaboost import Adaboost
from Models.DecisionTree import DecisionTree
from Models.ExtraTree import ExtraTree
from Models.MLP import MLP
from Models.NaiveBayes import NaiveBayes
from Models.RandomForest import RandomForest
from Utils.LoggerUtil import LoggerUtil


class Model:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.read_data = ReadData()
        self.bayes = NaiveBayes()
        self.decision_tree = DecisionTree()
        self.random_forest = RandomForest()
        self.extra_tree = ExtraTree()
        self.adaboost = Adaboost()
        self.mlp = MLP()

    @staticmethod
    def split_data(df, labels):
        x_train, x_test, y_train, y_test = train_test_split(df, labels, stratify=labels)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def vectorize_data(x_train, x_test, x_validation):
        """
        While we needed subject to estimate if the mail contains action item or not,
        we don't need it for model selection.
        :return:
        """
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 5),
            max_features=30000
        )

        x_train_features = vectorizer.fit_transform(x_train)
        x_test_features = vectorizer.fit_transform(x_test)
        x_validation_features = vectorizer.transform(x_validation)
        return x_train_features, x_test_features, x_validation_features

    def main(self):
        df, labels = self.read_data.main(n_rows=1000)
        x_train, x_validation, y_train, y_validation = self.split_data(df['message'], labels)
        x_train, x_test, y_train, y_test = self.split_data(x_train, y_train)
        x_train_features, x_test_features, x_validation_features = self.vectorize_data(x_train, x_test, x_validation)
        self.bayes.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                        x_validation=x_validation_features, y_validation=y_validation)
        self.decision_tree.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                                x_validation=x_validation_features, y_validation=y_validation)
        self.random_forest.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                                x_validation=x_validation_features, y_validation=y_validation)
        self.extra_tree.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                             x_validation=x_validation_features, y_validation=y_validation)
        self.adaboost.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                           x_validation=x_validation_features, y_validation=y_validation)
        self.mlp.main(x_train=x_train_features, y_train=y_train, x_test=x_test_features, y_test=y_test,
                      x_validation=x_validation_features, y_validation=y_validation)


if __name__ == '__main__':
    model = Model()
    model.main()
