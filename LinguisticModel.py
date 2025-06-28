from numpy import zeros
from sklearn.externals import joblib
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
        x_train, x_test, y_train, y_test = train_test_split(
            df, labels, stratify=labels, shuffle=True
        )
        return x_train, x_test, y_train, y_test

    @staticmethod
    def custom_word_tokenizer(tokens):
        return tokens

    def vectorize_data(self, data):
        """
        While we needed subject to estimate if the mail contains action item or not,
        we don't need it for model selection.
        :return:
        """
        vectorizer = TfidfVectorizer(
            tokenizer=self.custom_word_tokenizer,
            sublinear_tf=True,
            analyzer="word",
            lowercase=False,
            max_features=2500,
        )

        vectorizer.fit(data)
        return vectorizer

    @staticmethod
    def get_tagged_dataset(df):
        return df[~df["labels"]]["message"].head(n=1250), zeros(1250, dtype=bool)

    def train_models(self, x_train_features, x_test_features, y_train, y_test):
        nb_model = self.bayes.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        lr_model = self.lr.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        dt_model = self.dt.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        rf_model = self.rf.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        et_model = self.et.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        adaboost_model = self.adaboost.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )
        mlp_model = self.mlp.main(
            x_train=x_train_features,
            y_train=y_train,
            x_test=x_test_features,
            y_test=y_test,
        )

        model_dict = {
            "naive_bayes": nb_model,
            "logistic_regression": lr_model,
            "decision_tree": dt_model,
            "random_forest": rf_model,
            "extra_tree": et_model,
            "adaboost": adaboost_model,
            "mlp": mlp_model,
        }
        return model_dict

    def save_models(self, model_dict, models_path, vectorizer):
        joblib.dump(vectorizer, models_path + "/" + "vectorizer.mdl")
        for model_name, stat_dict in model_dict.items():
            if model_name in [
                "naive_bayes",
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "extra_tree",
                "adaboost",
                "mlp",
            ]:
                joblib.dump(stat_dict["model"], models_path + "/" + model_name + ".mdl")
            else:
                self.log.error("Non-Standard Model referenced")

    def get_best_model(self, model_dict):
        best_acc = 0
        best_model_name = ""
        for model_name, model_stat in model_dict.items():
            for key, value in model_stat.items():
                if key == "metrics":
                    accuracy = value["accuracy"]
                    if accuracy > best_acc:
                        best_model_name = model_name

        self.log.info("Best model determined is : {}".format(best_model_name))
        return model_dict[best_model_name]["model"]

    def main(self):
        models_path = self.config["models_path"]
        untagged_data_df = self.read_data.prepare_data(n_rows=3000)
        untagged_data, untagged_labels = (
            untagged_data_df["message"],
            untagged_data_df["labels"],
        )

        vectorizer = self.vectorize_data(untagged_data)
        x_train, x_test, y_train, y_test = self.split_data(
            untagged_data, untagged_labels
        )
        x_train_features = vectorizer.transform(x_train)
        x_test_features = vectorizer.transform(x_test)

        model_dict = self.train_models(
            x_train_features, x_test_features, y_train, y_test
        )
        self.save_models(
            model_dict=model_dict, models_path=models_path, vectorizer=vectorizer
        )
        best_model = self.get_best_model(model_dict)
        return best_model, vectorizer


if __name__ == "__main__":
    model = LinguisticModel()
    model.main()
