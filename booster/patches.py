
from distutils.version import LooseVersion
import sklearn
from sklearn.ensemble.gradient_boosting import QuantileLossFunction,\
    ExponentialLoss
import numpy as np
from six import PY2


# Patch over bug in scikit-learn (issue #9539)
if LooseVersion(sklearn.__version__) <= LooseVersion('0.18.2'):
    def __call__(self, y, pred, sample_weight=None):
        pred = pred.ravel()
        diff = y - pred
        alpha = self.alpha
    
        mask = y > pred
        if sample_weight is None:
            loss = (alpha * diff[mask].sum() -
                    (1.0 - alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            loss = ((alpha * np.sum(sample_weight[mask] * diff[mask]) -
                    (1.0 - alpha) * np.sum(sample_weight[~mask] * diff[~mask])) /
                    sample_weight.sum())
        return loss
    if PY2:
        from types import MethodType
        QuantileLossFunction.__call__ = MethodType(__call__, None, QuantileLossFunction)
    else:
        QuantileLossFunction.__call__ = __call__

# Patch over bug in scikit-learn (issue #9666)
if LooseVersion(sklearn.__version__) <= LooseVersion('0.19.1'):
    def negative_gradient(self, y, pred, **kargs):
        y_ = -(2. * y - 1.)
        return - y_ * np.exp(y_ * pred.ravel())
    if PY2:
        ExponentialLoss.negative_gradient = MethodType(negative_gradient, None, ExponentialLoss)
    else:
        ExponentialLoss.negative_gradient = negative_gradient

