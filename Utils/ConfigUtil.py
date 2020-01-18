import json


class ConfigUtil:
    def __init__(self):
        pass

    @staticmethod
    def get_config_instance():
        with open('/home/satwik/Documents/Hiring/huddl_assignment/Config.json') as json_data_file:
            data = json.load(json_data_file)

        return data
