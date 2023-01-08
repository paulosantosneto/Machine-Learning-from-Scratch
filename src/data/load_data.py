import json
import os

class Data(object):

    def __init__(self):
        self.data = {}

    def load_json(self):
        
        self.data["linear_regression"] = json.load(open("linear_regression_simple_example.json"))


