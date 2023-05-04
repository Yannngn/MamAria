from typing import Literal

import numpy as np
from jax._src.config import config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss
from tqdm.auto import tqdm

from calibrators.minibatch_multinomial import MiniBatchMultinomialRegression
from calibrators.utils import clip_for_log

config.update("jax_enable_x64", True)


class MiniBatchFullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
        weights_init: np.ndarray | None = None,
        initializer: Literal["identity"] | None = "identity",
        reg_norm: bool = False,
        ref_row: bool = True,
        optimizer: Literal["auto", "newton", "fmin_l_bfgs_b"] = "auto",
        batch_size: int = 1 * 360 * 600,
        max_iter: int = 1000,
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *args,
        **kwargs,
    ):
        self.weights_ = self.weights_init
        n_samples = X.shape[0]
        n_batches = n_samples // self.batch_size
        final_loss = []

        if X_val is None or y_val is None:
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
            optimizer=self.optimizer,  # type: ignore
        )

        main_loop = tqdm(
            range(self.max_iter), desc=f"iter 0 of {self.max_iter}", postfix={"log_loss": 0.0}, position=0
        )
        for idx in main_loop:
            iter_loss = []
            main_loop.set_description_str(f"iter {idx} of {self.max_iter}")

            iter_loop = tqdm(
                range(n_batches),
                desc=f"Mini Batch 0 of {n_batches}",
                postfix={"log_loss": 0.0},
                position=1,
                leave=False,
            )
            for jdx in iter_loop:
                iter_loop.set_description_str(f"Mini Batch {jdx} of {n_batches}")

                start_idx = jdx * self.batch_size
                end_idx = (jdx + 1) * self.batch_size

                if end_idx > n_samples:
                    break

                self.calibrator_.partial_fit(
                    X[start_idx:end_idx],
                    y[start_idx:end_idx],
                    *args,
                    **kwargs,
                )

                # valida mesmo com todos do _X_val e y_val?
                step_loss = log_loss(y_val, self.calibrator_.predict_proba(X_val))

                iter_loop.set_postfix(log_loss=round(step_loss, 3))
                iter_loss.append(step_loss)

            iter_loss = np.mean(iter_loss)
            final_loss.append(iter_loss)

            main_loop.set_postfix(log_loss=round(iter_loss, 3))
        main_loop.set_postfix(log_loss=round(np.mean(final_loss), 3))

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
