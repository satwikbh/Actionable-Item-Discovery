from sklearn.naive_bayes import BernoulliNB

from Models.Metrics import Metrics
from Utils.ConfigUtil import ConfigUtil
from Utils.Helper import Helper
from Utils.LoggerUtil import LoggerUtil


class NaiveBayes:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = Helper()
        self.metrics = Metrics()

    @staticmethod
    def train_model(x_train, y_train):
        model = BernoulliNB()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def test_model(model, x_test):
        return model.predict(x_test)

    def main(self, x_train, x_test, x_validation, y_train, y_test, y_validation):
        image_path = self.config["image_path"]
        model = self.train_model(x_train, y_train)
        self.log.info("{}\t{}\t{}".format(x_train.shape, x_test.shape, x_validation.shape))

        self.log.info("{} Model performance on test data".format(self.__class__.__name__))
        y_pred = self.test_model(model, x_test)
        cnf_matrix = self.metrics.metrics(y_true=y_test, y_predicted=y_pred)
        self.helper.plot_save_cnf_matrix(cnf_matrix, flag="test", model_name=self.__class__.__name__,
                                         image_path=image_path)

        self.log.info("{} Model performance on Validation data".format(self.__class__.__name__))
        y_pred = self.test_model(model, x_validation)
        cnf_matrix = self.metrics.metrics(y_true=y_validation, y_predicted=y_pred)

        self.helper.plot_save_cnf_matrix(cnf_matrix, flag="validation", model_name=self.__class__.__name__,
                                         image_path=image_path)
