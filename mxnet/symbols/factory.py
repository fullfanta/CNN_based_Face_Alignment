import os
import json
import importlib


SUPPORT_MODELS = ('vgg_16_reduced', 'inception-bn')


class SymbolFactory:
    def __init__(self, params):
        self.model = params
        if self.model in SUPPORT_MODELS:
            self.module = importlib.import_module('symbols.' + self.model)
        else:
            raise NotImplementedError

    def get_symbol(self):
        return self.module.get_symbol()
