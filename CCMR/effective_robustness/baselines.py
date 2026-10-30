
import json
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1/(1+np.exp(-x))


class Baseline:

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class BaselineIdentity(Baseline):

    def fit(self, X, y):
        xx = np.array([0, 1]).reshape((-1, 1))
        yy = np.array([0, 1]).reshape((-1, 1))
        self.reg = LinearRegression()
        self.reg.fit(xx, yy)
        pass

    def predict(self, X):
        self.fit(None, None)
        xx = np.array(X).reshape((-1, 1))
        return self.reg.predict(xx).reshape(-1)

    def score(self, xx, yy):
        xx = np.array(xx).reshape((-1, 1))
        yy = np.array(yy).reshape((-1, 1))
        return self.reg.score(xx, yy)


class BaselineLinearRegression(Baseline):

    def fit(self, X, y):
        xx = np.array(X).reshape((-1, 1))
        yy = np.array(y).reshape((-1, 1))
        self.reg = LinearRegression()
        self.reg.fit(xx, yy)
        pass

    def predict(self, X):
        xx = np.array(X).reshape((-1, 1))
        return self.reg.predict(xx).reshape(-1)

    def score(self, xx, yy):
        xx = np.array(xx).reshape((-1, 1))
        yy = np.array(yy).reshape((-1, 1))
        return self.reg.score(xx, yy)


class BaselineLogitLinearRegression(Baseline):

    def fit(self, xx, yy):
        self.xx = list(xx)
        self.yy = list(yy)
        logit_xx = _logit(np.array(xx).reshape((-1, 1)))
        logit_yy = _logit(np.array(yy).reshape((-1, 1)))

        self.reg = LinearRegression()
        self.reg.fit(logit_xx, logit_yy)
        pass

    def predict(self, xx):
        xx = _logit(np.array(xx).reshape((-1, 1)))
        return _expit(self.reg.predict(xx).reshape(-1))

    def score(self, xx, yy):
        xx = _logit(np.array(xx).reshape((-1, 1)))
        yy = _logit(np.array(yy).reshape((-1, 1)))
        return self.reg.score(xx, yy)

    def pretty_format(self):
        return f'expit( {self.reg.coef_[0, 0]} * logit( wauc_ID ) + {self.reg.intercept_[0]} )'

    def serialize_json(self, output_filename=None):
        d = {
            'class': str(self.__class__.__name__),
            'xx': list(self.xx),
            'yy': list(self.yy),
            'coeff_a': self.reg.coef_[0, 0],
            'coeff_b': self.reg.intercept_[0],
        }
        if output_filename:
            with open(output_filename, 'w') as f:
                json.dump(d, f, indent=3)
        return d

    @staticmethod
    def deserialize_json(filename):
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        assert json_dict['class'] == str(BaselineLogitLinearRegression.__name__), json_dict['class']
        assert len(json_dict['xx']) == len(json_dict['yy']), json_dict
        bl = BaselineLogitLinearRegression()
        bl.fit(json_dict['xx'], json_dict['yy'])
        
        return bl

        

class BaselineLogLinearRegression(Baseline):

    def fit(self, xx, yy):
        xx = np.log(np.array(xx).reshape((-1, 1)))
        yy = np.log(np.array(yy).reshape((-1, 1)))

        self.reg = LinearRegression()
        self.reg.fit(xx, yy)
        pass

    def predict(self, xx):
        xx = np.log(np.array(xx).reshape((-1, 1)))
        return np.exp(self.reg.predict(xx).reshape(-1))

    def score(self, xx, yy):
        xx = np.log(np.array(xx).reshape((-1, 1)))
        yy = np.log(np.array(yy).reshape((-1, 1)))
        return self.reg.score(xx, yy)
