import logging
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
from jax._src.config import config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from calibrators.utils import clip_for_log, clip_jax

config.update("jax_enable_x64", True)


def _get_weights(params, k: int, ref_row: bool) -> jax.Array:
    """Reshapes the given params (weights) into the full matrix including 0"""

    raw_weights = jnp.reshape(params, (-1, k + 1))

    if ref_row:
        return raw_weights - jnp.repeat(jnp.reshape(raw_weights[-1, :], (1, -1)), k, axis=0)

    return raw_weights


def _softmax(X: jax.Array) -> jax.Array:
    """Compute the softmax of matrix X in a numerically stable way."""
    X = X - jnp.max(X, axis=1, keepdims=True)
    exps = jnp.exp(X)

    return exps / jnp.sum(exps, axis=1, keepdims=True)


def _calculate_outputs(weights: jax.Array, X: jax.Array) -> jax.Array:
    assert (
        weights.transpose().shape[0] == X.shape[1]
    ), f"{weights.transpose().shape[0]=} and {X.shape[1]=} should be equal"
    return _softmax(jnp.dot(X, weights.transpose()))


class NewtonUpdate(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k: int,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
        ref_row: bool = True,
        maxiter: int = 1024,
        ftol: float = 1e-12,
        gtol: float = 1e-8,
    ):
        self.k = k
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.ref_row = ref_row
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol

    def fit(self, weights: jax.Array, X: jax.Array, y: jax.Array):
        L_list = [self.loss_fn(weights, X, y)]

        newton_loop = tqdm(
            range(self.maxiter), position=2, leave=False, desc="_newton_update: ", postfix={"loss": 0.0}
        )

        for idx in newton_loop:
            gradient = jax.grad(self.loss_fn)(weights, X, y)

            if jnp.sum(jnp.abs(gradient)) < self.gtol:
                logging.debug(f"{jnp.sum(jnp.abs(gradient))=} < {self.gtol=}")
                break

            # FIXME hessian is ocasionally NaN
            hessian = jax.hessian(self.loss_fn)(weights, X, y)

            try:
                inverse_hessian = scipy.linalg.pinv(hessian)
                updates: jax.Array = jnp.matmul(inverse_hessian, gradient)
            except (np.linalg.LinAlgError, ValueError) as err:
                logging.error(err)
                updates: jax.Array = gradient

            L, temp_weights = self._step(weights, updates, X, y, L_list[-1])
            L_list.append(L)

            logging.debug(
                f"_newton_update: after {idx} iterations log-loss = {L:.7e}, sum_grad = {jnp.sum(jnp.abs(gradient)):.7e}"
            )

            if jnp.isnan(L):
                logging.error(f"_newton_update: log-loss is NaN")
                break

            elif np.diff(L_list[-2:]) > 0:
                logging.debug(f"_newton_update: Terminate as the loss increased {np.diff(L_list[-2:])}.")
                break

            elif idx >= 5:
                if np.abs(np.min(np.diff(L_list[-5:]))) < self.ftol:
                    logging.debug(f"_newton_update: Terminate as there is not enough changes on loss.")
                    weights = jnp.copy(temp_weights)
                    break
            else:
                weights = jnp.copy(temp_weights)

            newton_loop.set_postfix(loss=L)

        L = self.loss_fn(weights, X, y)

        logging.debug(
            f"_newton_update: after {idx} iterations final log-loss = {L:.7e}, sum_grad = {jnp.abs(gradient).sum():.7e}"
        )

        self.weights = _get_weights(weights, self.k, self.ref_row)

        return self

    def _step(
        self, weights: jax.Array, updates: jax.Array, X: jax.Array, y: jax.Array, last_loss: float
    ) -> Tuple[float, jax.Array]:
        for step_size in jnp.hstack((jnp.linspace(1, 0.1, 10), jnp.logspace(-2, -32, 31))):
            step_weights = weights - jnp.ravel(updates * step_size)

            if jnp.any(jnp.isnan(step_weights)):
                logging.error(f"_newton_update: There are NaNs in step_weights")

            L = self.loss_fn(step_weights, X, y)

            # usa o maior step que gerar loss menor
            if L < last_loss:
                break

        return L, step_weights

    def loss_fn(self, weights: jax.Array, X: jax.Array, y: jax.Array) -> float:
        weights = _get_weights(weights, self.k, self.ref_row)

        outputs = clip_jax(_calculate_outputs(weights, X))
        loss = jnp.mean(-jnp.log(jnp.sum(y * outputs, axis=1)))

        if self.reg_mu is None:
            reg = jnp.hstack([jnp.eye(self.k), jnp.zeros((self.k, 1))])
            return loss + self.reg_lambda * jnp.sum((weights - reg) ** 2)

        weights_hat = weights - jnp.hstack([weights[:, :-1] * jnp.eye(self.k), jnp.zeros((self.k, 1))])
        loss = (
            loss + self.reg_lambda * jnp.sum(weights_hat[:, :-1] ** 2) + self.reg_mu * jnp.sum(weights_hat[:, -1] ** 2)
        )

        return loss.astype(float)

    def predict_proba(self, S: jax.Array) -> jax.Array:
        S_ = jnp.hstack((S, jnp.ones((len(S), 1))))
        return _softmax(jnp.dot(S_, self.weights.transpose()))

    def predict(self, S: jax.Array) -> jax.Array:
        return self.predict_proba(S)


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        weights_init: jax.Array | None = None,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
        reg_norm: bool = False,
        ref_row: bool = True,
    ):
        self.weights_init = weights_init
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.reg_norm = reg_norm
        self.ref_row = ref_row

    def __setup(self, y):
        first_fit = not hasattr(self, "classes")
        # não ter o attr classes
        if not first_fit:
            self.ravel_weights = jnp.ravel(self.weights_)
            return

        logging.info("setting up first_fit")

        self.classes: np.ndarray = np.unique(y)
        self.k = len(self.classes)

        if self.k > 36:
            raise ValueError("To many classes for this implemention")

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (self.k * (self.k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (self.k * (self.k - 1))
                self.reg_mu = self.reg_mu / self.k

        self.ravel_weights = self._get_initial_weights()
        self.newton_update = NewtonUpdate(self.k, self.reg_lambda, self.reg_mu, self.ref_row)

    @property
    def coef_(self):
        return self.weights_[:, :-1]

    @property
    def intercept_(self):
        return self.weights_[:, -1]

    def predict_proba(self, S):
        S_ = jnp.hstack((S, jnp.ones((len(S), 1))))

        return jnp.asarray(_calculate_outputs(self.weights_, S_))

    # FIXME Should we change predict for the argmax?
    def predict(self, S):
        return self.predict_proba(S)

    def partial_fit(self, X, y):
        self.__setup(y)

        X = jnp.hstack((X, jnp.ones((len(X), 1))))

        y = label_binarize(y, classes=self.classes)

        if self.k == 2:
            y = jnp.hstack([1 - y, y])

        self.weights_ = self.newton_update.fit(self.ravel_weights, X, y).weights

        return self

    def fit(self, X, y):
        self.classes: np.ndarray = np.unique(y)
        self.k = len(self.classes)

        if self.k > 36:
            raise ValueError("To many classes for this implemention")

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (self.k * (self.k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (self.k * (self.k - 1))
                self.reg_mu = self.reg_mu / self.k

        self.ravel_weights = self._get_initial_weights()
        self.newton_update = NewtonUpdate(self.k, self.reg_lambda, self.reg_mu, self.ref_row)

        X = jnp.hstack((X, jnp.ones((len(X), 1))))

        y = label_binarize(y, classes=self.classes)

        if self.k == 2:
            y = jnp.hstack([1 - y, y])

        weights = self.newton_update.fit(X, y, self.k)

        self.ravel_weights = _get_weights(weights, self.k, self.ref_row)

        return self

    def _get_initial_weights(self):
        if self.weights_init is None:
            raw_weights = np.zeros((self.k, self.k + 1)) + np.hstack([np.eye(self.k), np.zeros((self.k, 1))])
            return np.ravel(raw_weights)

        return self.weights_init


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
        weights_init: np.ndarray | None = None,
        reg_norm: bool = False,
        ref_row: bool = True,
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
        self.weights_init = weights_init  # Input weights for initialization
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.batch_size = batch_size
        self.max_iter = max_iter

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
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
        self.calibrator_ = MultinomialRegression(
            reg_lambda=self.reg_lambda,
            reg_mu=self.reg_mu,
            reg_norm=self.reg_norm,
            ref_row=self.ref_row,
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

                self.calibrator_.partial_fit(X[start_idx:end_idx], y[start_idx:end_idx])

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
