from sklearn.tree import ExtraTreeClassifier

from Models.Metrics import Metrics
from Utils.ConfigUtil import ConfigUtil
from Utils.Helper import Helper
from Utils.LoggerUtil import LoggerUtil


class ExtraTree:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.helper = Helper()
        self.metrics = Metrics()

    @staticmethod
    def train_model(x_train, y_train):
        model = ExtraTreeClassifier()
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def test_model(model, x_test):
        return model.predict(x_test)

    def main(self, x_train, x_test, y_train, y_test):
        image_path = self.config["image_path"]
        model = self.train_model(x_train, y_train)

        self.log.info("{} Model performance on test data".format(self.__class__.__name__))
        y_pred = self.test_model(model, x_test)
        acc_score, cr_report, cnf_matrix = self.metrics.metrics(y_true=y_test, y_predicted=y_pred)
        self.helper.plot_save_cnf_matrix(cnf_matrix, flag="test", model_name=self.__class__.__name__,
                                         image_path=image_path)
        return {
            "model": model,
            "metrics": {
                "accuracy": acc_score,
                "classification_report": cr_report,
                "confusion_matrix": cnf_matrix
            }
        }
