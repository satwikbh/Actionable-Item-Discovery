from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from Utils.LoggerUtil import LoggerUtil


class DecisionTree:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def train_model(x_train, y_train):
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def test_model(model, x_test):
        return model.predict(x_test)

    @staticmethod
    def plot_save_cnf_matrix(cnf_matrix):
        pass

    def metrics(self, y_test, y_pred):
        precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred)
        cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        self.plot_save_cnf_matrix(cnf_matrix)

    def main(self, x_train, x_test, y_train, y_test):
        model = self.train_model(x_train, y_train)
        y_pred = self.test_model(model, x_test)
        self.metrics(y_test, y_pred)
