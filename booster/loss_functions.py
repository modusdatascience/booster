from sklearn.ensemble.gradient_boosting import RegressionLossFunction, QuantileEstimator
import numpy as np

def log_one_plus_exp_x(x):
    lower = -10.
    upper = 35.
    result = np.zeros_like(x)
    low_idx = x < lower
    result[low_idx] = np.exp(x[low_idx])
    high_idx = x > upper
    result[high_idx] = x[high_idx]
    middle_idx = ~(low_idx | high_idx)
    result[middle_idx] = np.log(1+np.exp(x[middle_idx]))
    return result

def one_over_one_plus_exp_x(x):
    lower = -100.
    upper = 100.
    result = np.zeros_like(x)
    low_idx = x < lower
    result[low_idx] = 1.
    high_idx = x > upper
    result[high_idx] = 0.
    middle_idx = ~(low_idx | high_idx)
    result[middle_idx] = 1. / (1. + np.exp(x[middle_idx]))
    return result
    
class SmoothQuantileLossFunction(RegressionLossFunction):
    def __init__(self, n_classes, tau, alpha):
        super(SmoothQuantileLossFunction, self).__init__(n_classes)
        self.tau = tau
        self.alpha = alpha
    
    def init_estimator(self):
        return QuantileEstimator(self.tau)
    
    def __call__(self, y, pred, sample_weight=None):
        x = y - pred
        if sample_weight is not None:
            return np.dot(sample_weight, (self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x)))
        else:
            return np.sum(self.tau * x + self.alpha * log_one_plus_exp_x(-(1./self.alpha)*x))
    
    def negative_gradient(self, y, pred, sample_weight=None):
        x =  y - pred
        if sample_weight is not None:
            return  sample_weight * (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        else:
            return (self.tau - one_over_one_plus_exp_x((1. / self.alpha) * x))
        
    def _update_terminal_region(self, *args, **kwargs):
        raise NotImplementedError()