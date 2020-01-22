from pandas import read_csv
from spacy import load
from spacy.lang.en.stop_words import STOP_WORDS

from Core.Logic import Logic
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class CheckPreTaggedData:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.logic = Logic()

    def main(self):
        nlp = load("en_core_web_sm")
        stop_words = set.union(STOP_WORDS, {'ect', 'hou', 'com', 'recipient', 'na', 'ou', 'cn', 'enron', 'zdnet'})
        tagged_df = read_csv(self.config["tagged_path"] + "/" + "actions.csv", header=None)
        tagged_df[0] = tagged_df[0].apply(
            lambda x: [item.lower().strip() for item in x.split() if item.lower().strip() not in stop_words]
        )
        tagged_df["labels"] = tagged_df[0].apply(lambda x: self.logic.apply_rules(x, nlp))
        self.log.info("Values detected by Model in Pre-Tagged Sentences : {}".format(tagged_df.labels.value_counts()))


if __name__ == '__main__':
    check = CheckPreTaggedData()
    check.main()
