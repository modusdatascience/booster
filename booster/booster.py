from sklearn.base import clone, BaseEstimator
from toolz.dicttoolz import valmap
from .line_search import golden_section_search, zoom_search, zoom
from operator import __sub__, __lt__, __add__
from toolz.itertoolz import sliding_window
from itertools import starmap
from toolz.functoolz import flip, curry
from sklearn.exceptions import NotFittedError
from sklearntoolsbase.sklearntoolsbase import growd, shrinkd,\
    fit_predict, LinearCombination, notnone
from six.moves import reduce

# This import patches over some bugs in scikit-learn
from . import patches  # @UnusedImport
from toolz.curried import valfilter
from sklearn2code.sym.base import sym_transform, sym_decision_function,\
    sym_predict, sym_score_to_decision, sym_score_to_proba, syms
from sklearn2code.utility import xlabels
import numpy as np
import time

def never_stop_early(**kwargs):
    return False

class NIterationsWithoutImprovementOverThreshold(object):
    def __init__(self, stat, n, threshold=0.):
        self.stat = stat
        self.n = n
        self.threshold = threshold
    
    def __call__(self, losses, **kwargs):
        if len(losses) <= self.n:
            return False
        return all(map(curry(__lt__)(-self.threshold), starmap(self.stat, sliding_window(2, losses[-(self.n+1):]))))

@curry
def stop_after_n_iterations_without_stat_improvement_over_threshold(stat, n, threshold=0.):
    return NIterationsWithoutImprovementOverThreshold(stat, n, threshold)

stop_after_n_iterations_without_improvement_over_threshold = stop_after_n_iterations_without_stat_improvement_over_threshold(flip(__sub__))

def percent_reduction(before, after):
    return 100*(after - before) / float(before)

stop_after_n_iterations_without_percent_improvement_over_threshold = stop_after_n_iterations_without_stat_improvement_over_threshold(percent_reduction)

class BoosterFitRecord(object):
    def __init__(self, losses, times, stopping_condition=None):
        self.losses = losses
        self.times = times
        self.stopping_condition = stopping_condition
    
    widths = (12, 12, 12)
    def strs_to_str(self, strs):
        return reduce(__add__, [s[:l] + ' ' * max(0, l-len(s)) + ' ' for s, l in zip(strs, self.widths)])
    
    def header_str(self):
        return self.strs_to_str(['Iteration', 'Time', 'Loss'])
    
    def line_to_str(self, line_num):
        iteration = repr(line_num)
        time = repr(self.times[line_num])
        loss = repr(self.losses[line_num])
        return self.strs_to_str([iteration, time, loss])
    
    def line_break_str(self):
        return '-' * (sum(self.widths) + len(self.widths))
    
    def to_str(self):
        return (self.line_break_str() +
                self.header_str() + '\n' + self.line_break_str() + 
                '\n'.join(map(self.line_to_str, range(len(self)))))
    
    def last_to_str(self):
        return self.line_to_str(len(self) - 1)
    
    def stopping_str(self):
        return self.stopping_condition
        
    def __len__(self):
        return len(self.losses)
    
class Booster(BaseEstimator):
    def __init__(self, base_estimator, loss_function, learning_rate=.1, n_estimators=100,
                 stopper=never_stop_early, verbose=0):
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.stopper = stopper
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None, exposure=None, previous_prediction=None):
        fit_args = {'X': growd(2,X)}
        if sample_weight is not None:
            fit_args['sample_weight'] = shrinkd(1, sample_weight)
        if exposure is not None:
            fit_args['exposure'] = shrinkd(1, exposure)
        if self._estimator_type == 'classifier':
            self.classes_, y_ = np.unique(growd(2,y), return_inverse=True)
        else:
            y_ = growd(2,y)
        fit_args['y'] = y_
        coefficients = []
        estimators = []
        times = []
        losses = []
        stopping_condition = 'All Steps Completed'
        previous_time = time.time()
        record = BoosterFitRecord(losses, times)
        if self.verbose >= 1:
            print(record.line_break_str())
            print(record.header_str())
            print(record.line_break_str())
        if previous_prediction is None:
            initial_estimator = self.loss_function.init_estimator()
            initial_estimator.fit(**fit_args)
            initial_estimator.xlabels_ = xlabels(X)
            coefficients.append(1.)
            estimators.append(initial_estimator)
        predict_args = {'X':X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        if previous_prediction is None:
            prediction = shrinkd(1, initial_estimator.predict(**valmap(shrinkd(1), predict_args)))
        else:
            prediction = previous_prediction.copy()
            
        gradient_args = {'y':y, 'pred':prediction}
        if sample_weight is not None:
            gradient_args['sample_weight': sample_weight]
        if exposure is not None:
            gradient_args['exposure': exposure]
        gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
        partial_arguments = {'y':y}
        if sample_weight is not None:
            partial_arguments['sample_weight'] = sample_weight
        if exposure is not None:
            partial_arguments['exposure'] = exposure
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
        self.initial_loss_ = loss_function(prediction)
        loss = self.initial_loss_
        losses.append(loss)
        t = time.time()
        times.append(t - previous_time)
        previous_time = t
        predict_args = {'X': X}
        if exposure is not None:
            predict_args['exposure'] = shrinkd(1, exposure)
        self.early_stop_ = False
        previous_time = time.time()
        if self.verbose >= 1:
            print(record.last_to_str())
        for iteration in range(self.n_estimators):
#             previous_loss = loss
#             if self.verbose >= 1:
#                 print('Fitting estimator %d...' % (iteration + 1))
            fit_args['y'] = shrinkd(1, gradient)
            estimator = clone(self.base_estimator)
            try:
                approx_gradient = shrinkd(1, fit_predict(estimator, **fit_args))
            except:
                raise
#             if self.verbose >= 1:
#                 print('Fitting for estimator %d complete.' % (iteration + 1))
#             if self.verbose >= 1:
#                 print('Computing alpha for estimator %d...' % (iteration + 1))
            alpha = zoom_search(golden_section_search(1e-16), zoom(1., 20, 2.), loss_function, prediction, approx_gradient)
            alpha *= self.learning_rate
#             if self.verbose >= 1:
#                 print('alpha = %f' % alpha)
#             if self.verbose >= 1:
#                 print('Computing alpha for estimator %d complete.' % (iteration + 1))
        
            prediction += alpha * approx_gradient
            loss = loss_function(prediction)
            coefficients.append(alpha)
            estimators.append(estimator)
            losses.append(loss)
            t = time.time()
            times.append(t - previous_time)
            previous_time = t
            if self.verbose >= 1:
                print(record.last_to_str())
            if self.stopper(iteration=iteration, coefficients=coefficients, losses=losses, 
                            gradient=gradient, approx_gradient=approx_gradient):#, approx_gradient_cv=approx_gradient_cv):
                self.early_stop_ = True
                stopping_condition = 'Early Stop'
                break
            gradient_args['pred'] = prediction
            gradient = shrinkd(1, self.loss_function.negative_gradient(**valmap(shrinkd(1), gradient_args)))
        self.coefficients_ = coefficients
        self.estimators_ = estimators
        self.losses_ = losses
        self.score_ = (self.initial_loss_ - loss) / self.initial_loss_
        self.estimator_ = LinearCombination(self.estimators_, self.coefficients_)
        record.stopping_condition = stopping_condition
        if self.verbose >= 1:
            print(record.stopping_str())
        return self
    
    def statistic_over_steps(self, X, y, statistic, exposure=None):
        result = []
        predict_args = {'X': X}
        if exposure is not None:
            predict_args['exposure'] = exposure
        for j in range(1, len(self.estimators_)+1):
            model = LinearCombination(self.estimators_[:j], self.coefficients_[:j])
            pred = model.predict(X)
            result.append(statistic(y, pred))
        return result

    def score(self, X, y, sample_weight=None, exposure=None):
        partial_arguments = {}# {'X': growd(2,X)}
        predict_arguments = {'X': growd(2, X)}
        if sample_weight is not None:
            partial_arguments['sample_weight'] = shrinkd(1, sample_weight)
        if exposure is not None:
            partial_arguments['exposure'] = shrinkd(1, exposure)
            predict_arguments['exposure'] = partial_arguments['exposure']
        if self._estimator_type == 'classifier':
            self.classes_, y_ = np.unique(growd(2,y), return_inverse=True)
        else:
            y_ = growd(2,y)
        partial_arguments['y'] = y_
#         partial_arguments = self._process_args(y=y, sample_weight=sample_weight, exposure=exposure)
#         predict_arguments = dict(X=X, exposure=partial_arguments['exposure'])
        loss_function = lambda pred: self.loss_function(pred=shrinkd(1, pred), **valmap(shrinkd(1), partial_arguments))
        prediction = shrinkd(1, self.predict(**predict_arguments))
        loss = loss_function(prediction)
        initial_prediction = shrinkd(1, self.coefficients_[0] * self.estimators_[0].predict(**predict_arguments))
        initial_loss = loss_function(initial_prediction)
        return (initial_loss - loss) / initial_loss
    
    def syms(self):
        return syms(self.estimator_)
    
    def transform(self, X, exposure=None):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        return self.estimator_.transform(X=X, exposure=exposure)
    
    def sym_transform(self):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        return sym_transform(self.estimator_)
    
    def predict(self, X, exposure=None):
        score = self.decision_function(X=X, exposure=exposure)
        if hasattr(self.loss_function, '_score_to_decision'):
            return self.loss_function._score_to_decision(score)
        else:
            return score
    
    def sym_predict(self):
        score = self.sym_decision_function()
        score_to_decision = sym_score_to_decision(self.loss_function)
        return score_to_decision.compose(score)
    
    def predict_proba(self, X, exposure=None):
        if not hasattr(self.loss_function, '_score_to_proba'):
            raise AttributeError()
        score = self.decision_function(X=X, exposure=exposure)
        return self.loss_function._score_to_proba(score)
    
    def sym_predict_proba(self):
        score = self.sym_decision_function()
        score_to_proba = sym_score_to_proba(self.loss_function)
        return score_to_proba.compose(score)
    
    def decision_function(self, X, exposure=None):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        pred_args = valmap(growd(2), valfilter(notnone, dict(X=X, exposure=exposure)))
        score = self.estimator_.predict(**pred_args)
        return score
    
    def sym_decision_function(self):
        if not hasattr(self, 'estimator_'):
            raise NotFittedError()
        score = sym_predict(self.estimator_)
        return score
    
    @property
    def _estimator_type(self):
        if hasattr(self.loss_function, '_score_to_decision'):
            return 'classifier'
        else:
            return 'regressor'
        