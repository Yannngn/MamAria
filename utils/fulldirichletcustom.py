from tkinter import N
import jax.numpy as np
import logging
import numpy as raw_np
import scipy

from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from dirichletcal.calib.multinomial import MultinomialRegression, _get_weights, _gradient, _newton_update, _objective
from dirichletcal.utils import clip_for_log
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import label_binarize

class FullDirichletCalibratorCustom(FullDirichletCalibrator):
    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):
        self.weights_ = self.weights_init

        k = np.shape(X)[1]
        
        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        self.calibrator_ = MultinomialRegression(method='Full',
                                                 reg_lambda=self.reg_lambda,
                                                 reg_mu=self.reg_mu,
                                                 reg_norm=self.reg_norm,
                                                 ref_row=self.ref_row,
                                                 optimizer=self.optimizer)
        
        cv = KFold(10)
        for train, _ in cv.split(_X):
            sli = len(train) // 10
            for i in range(10): 
                self.calibrator_.fit(_X[train[i*sli:(i+1)*sli]], y[train[i*sli:(i+1)*sli]], *args, **kwargs)
            
        # self.calibrator_.fit(_X, y, *args, **kwargs)
        final_loss = log_loss(y_val, self.calibrator_.predict_proba(_X_val))

        return self
    
class MultinomialRegressionCustom(MultinomialRegression):
    def fit(self, X, y, *args, **kwargs):
        self.__setup()

        X_ = np.hstack((X, np.ones((len(X), 1))))

        self.classes = raw_np.unique(y)

        k = len(self.classes)

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (k * (k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (k * (k - 1))
                self.reg_mu = self.reg_mu / k

        target = label_binarize(y, classes=self.classes)

        if k == 2:
            target = np.hstack([1-target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * np.hstack([X_]*m)).reshape((n, m, m))

        logging.debug(self.method)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if (self.optimizer == 'newton'
            or (self.optimizer == 'auto' and k <= 36)):
            weights = _newton_update(self.weights_0_, X_, XXT, target, k,
                                     self.method, reg_lambda=self.reg_lambda,
                                     reg_mu=self.reg_mu, ref_row=self.ref_row,
                                     initializer=self.initializer,
                                     reg_format=self.reg_format)
        elif (self.optimizer == 'fmin_l_bfgs_b'
              or (self.optimizer == 'auto' and k > 36)):

            _gradient_np = lambda *args, **kwargs: raw_np.array(_gradient(*args, **kwargs))

            res = scipy.optimize.fmin_l_bfgs_b(func=_objective,
                                               fprime=_gradient_np,
                                               x0=self.weights_0_,
                                               args=(X_, XXT, target, k,
                                                     self.method,
                                                     self.reg_lambda,
                                                     self.reg_mu, self.ref_row,
                                                     self.initializer,
                                                     self.reg_format),
                                               maxls=128,
                                               factr=1.0)
            weights = res[0]
        else:
            raise(ValueError('Unknown optimizer: {}'.format(self.optimizer)))

        self.weights_ = _get_weights(weights, k, self.ref_row, self.method)

        return self