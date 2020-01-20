from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Metrics:
    def __init__(self):
        self.config = ConfigUtil.get_config_instance()
        self.log = LoggerUtil(self.__class__.__name__).get()

    def metrics(self, y_true, y_predicted):
        precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true=y_true, y_pred=y_predicted)
        self.log.info("Precision: {}\tRecall: {}\tFScore: {}\tTrueSum: {}".format(precision, recall, f_score, true_sum))
        cnf_matrix = confusion_matrix(y_true=y_true, y_pred=y_predicted)
        return cnf_matrix
