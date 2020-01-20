import matplotlib.pyplot as plt

from pandas import DataFrame
from seaborn import heatmap, set


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def is_list_not_empty(l):
        return len(l) != 0

    @staticmethod
    def is_list_empty(l):
        return len(l) == 0

    @staticmethod
    def plot_save_cnf_matrix(cnf_matrix, model_name, flag, image_path):
        plt.figure()
        df = DataFrame(cnf_matrix, range(2), range(2))
        set(font_scale=1.4)
        heatmap(df, annot=True, annot_kws={"size": 14})
        plt.title('Confusion Matrix for {} Model'.format(model_name))
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig(image_path + "/" + "confusion_matrix_" + str(model_name) + "_" + str(flag) + ".png")
