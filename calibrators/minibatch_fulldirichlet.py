# import logging

import jax.numpy as jnp
import numpy as np
from jax.config import config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss
from tqdm.auto import tqdm

from calibrators.minibatch_multinomial import MiniBatchMultinomialRegression

config.update("jax_enable_x64", True)


def clip_for_log(X):
    eps = jnp.finfo(X.dtype).tiny
    return jnp.clip(X, eps, 1 - eps)


class MiniBatchFullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        reg_lambda=0.0,
        reg_mu=None,
        weights_init=None,
        initializer="identity",
        reg_norm=False,
        ref_row=True,
        optimizer="auto",
        batch_size=1 * 360 * 600,
        max_iter=1000,
    ):
        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None
            then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            comp_l2: (bool) If true, then complementary L2 regularization used
            (off-diagonal regularization)
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b')
                If 'auto': then 'newton' for less than 37 classes and
                fmin_l_bfgs_b otherwise
                If 'newton' then uses our implementation of a Newton method
                If 'fmin_l_bfgs_b' then uses scipy.ptimize.fmin_l_bfgs_b which
                implements a quasi Newton method
        """
        self.reg_lambda = reg_lambda
        # Complementary L2 regularization. (Off-diagonal regularization)
        self.reg_mu = reg_mu
        self.weights_init = weights_init  # Input weights for initialisation
        self.initializer = initializer
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):
        self.weights_ = self.weights_init
        n_samples = X.shape[0]
        n_batches = n_samples // self.batch_size
        final_loss = []

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        X = np.log(clip_for_log(X))
        X_val = np.log(clip_for_log(X_val))

        # intanciar o calibrador no init?
        self.calibrator_ = MiniBatchMultinomialRegression(
            method="Full",
            reg_lambda=self.reg_lambda,
            reg_mu=self.reg_mu,
            reg_norm=self.reg_norm,
            ref_row=self.ref_row,
            optimizer=self.optimizer,
        )

        main_loop = tqdm(
            range(self.max_iter), desc="Fold log_loss: 0.0", position=0
        )
        for _ in main_loop:
            iter_loss = []

            iter_loop = tqdm(
                range(n_batches),
                desc="Mini Batch log_loss: 0.0",
                position=1,
                leave=False,
            )
            for idx in iter_loop:
                start_idx = idx * self.batch_size
                end_idx = (idx + 1) * self.batch_size
                if end_idx > n_samples:
                    break

                self.calibrator_.partial_fit(
                    X[start_idx:end_idx],
                    y[start_idx:end_idx],
                    *args,
                    **kwargs,
                )

                # valida mesmo com todos do _X_val e y_val?
                step_loss = log_loss(
                    y_val, self.calibrator_.predict_proba(X_val)
                )

                iter_loop.set_description_str(
                    f"Mini Batch log_loss: {step_loss:.3f}"
                )
                iter_loss.append(step_loss)

            iter_loss = np.mean(iter_loss)
            final_loss.append(iter_loss)

            iter_loop.close()
            main_loop.set_description_str(f"Fold log_loss: {iter_loss:.3f}")

        main_loop.set_description_str(
            f"Fold final log_loss: {np.mean(final_loss):.3f}"
        )
        main_loop.close()

        return self

    @property
    def weights(self):
        if self.calibrator_ is not None:
            return self.calibrator_.weights_
        return self.weights_init

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict_proba(S))

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict(S))
