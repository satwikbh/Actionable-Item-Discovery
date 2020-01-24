from sklearn.externals import joblib
from spacy import load

from Core.Logic import Logic
from Core.ReadData import ReadData
from LinguisticModel import LinguisticModel
from Models.Metrics import Metrics
from Utils.ConfigUtil import ConfigUtil
from Utils.Helper import Helper
from Utils.LoggerUtil import LoggerUtil


class TestModel:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.model = LinguisticModel()
        self.read_data = ReadData()
        self.metrics = Metrics()
        self.helper = Helper()
        self.logic = Logic()

    def check_if_trained(self):
        models_path = self.config["models_path"]
        model = joblib.load(models_path + "/" + "mlp.mdl")
        vectorizer = joblib.load(models_path + "/" + "vectorizer.mdl")
        return vectorizer, model

    def main(self, test=False):
        if test:
            nlp = load("en_core_web_sm")
            vectorizer, model = self.check_if_trained()
            self.log.info("Please enter the sentence")
            sentence = str(input())
            tokens = self.read_data.transform_sentence(sentence)
            features = vectorizer.transform(tokens)
            predictions = model.predict(features)
            ling_pred = self.logic.apply_rules(text_tokens=tokens, nlp=nlp)
            self.log.info("Given sentence : {}".format(sentence))
            self.log.info("Prediction of Linguistic Model : {}".format(ling_pred))
            self.log.info("Prediction of ML Model : {}".format(any(predictions)))
        else:
            model, vectorizer = self.model.main()
            tagged_data_df = self.read_data.prepare_tagged_data()
            features = vectorizer.transform(tagged_data_df["data"])
            labels = tagged_data_df["labels"]
            predictions = model.predict(features)

            acc_score, cr_report, cnf_matrix = self.metrics.metrics(y_true=labels, y_predicted=predictions)
            self.helper.plot_save_cnf_matrix(cnf_matrix=cnf_matrix, model_name="satwik", flag="test",
                                             image_path="/home/satwik/Documents/Hiring/huddl_assignment/Images/")


if __name__ == '__main__':
    test = TestModel()
    test.main(test=True)
