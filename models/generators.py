import torch
from models.base import BaseGenerator
from pysr import PySRRegressor

class PySR(BaseGenerator):
    """
    Wrapper for the PySR symbolic regression library.
    """
    def __init__(self, kwargs):
        super(PySR, self).__init__()
        self.model = PySRRegressor(**kwargs)

    def fit(self, X, y):
        self.pareto = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)