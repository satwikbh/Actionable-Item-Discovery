from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Metrics:
    def __init__(self):
        self.config = ConfigUtil.get_config_instance()
        self.log = LoggerUtil(self.__class__.__name__).get()

    def metrics(self, y_true, y_predicted):
        cr_report = classification_report(y_true=y_true, y_pred=y_predicted)
        acc_score = accuracy_score(y_true=y_true, y_pred=y_predicted)
        self.log.info("Accuracy Score : {}".format(acc_score))
        self.log.info("Classification Report: \n{}".format(cr_report))
        cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted)
        return cnf_matrix
