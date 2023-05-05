from __future__ import division

import logging
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
from jax._src.config import config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from calibrators.utils import clip_jax

config.update("jax_enable_x64", True)


class MiniBatchMultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        weights_0: Any = None,
        method: Literal["Full", "Diag", "FixDiag"] = "Full",
        initializer: Literal["identity"] | None = "identity",
        reg_format: Literal["identity"] | None = None,
        reg_lambda: float = 0.0,
        reg_mu: float | None = None,
        reg_norm: bool = False,
        ref_row: bool = True,
        optimizer: Literal["auto", "newton", "fmin_l_bfgs_b"] = "auto",
    ):
        """
        Params:
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b')
                If 'auto': then 'newton' for less than 37 classes and
                fmin_l_bfgs_b otherwise
                If 'newton' then uses our implementation of a Newton method
                If 'fmin_l_bfgs_b' then uses scipy.ptimize.fmin_l_bfgs_b which
                implements a quasi Newton method
        """
        if method not in ["Full", "Diag", "FixDiag"]:
            raise ValueError(f"method {method} not avaliable")

        self.weights_0 = weights_0
        self.method = method
        self.initializer = initializer
        self.reg_format = reg_format
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # If number, then ODIR is applied
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer

    def __setup(self, y):
        first_fit = not hasattr(self, "classes")
        # não ter o attr classes
        if not first_fit:
            self.weights_0_ = self.weights_
            return
        logging.info("setting up first_fit")
        self.classes: np.ndarray = np.unique(y)
        self.weights_ = self.weights_0
        self.weights_0_ = self.weights_0

        self.num_classes = len(self.classes)

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (self.num_classes * (self.num_classes + 1))
            else:
                self.reg_lambda = self.reg_lambda / (self.num_classes * (self.num_classes - 1))
                self.reg_mu = self.reg_mu / self.num_classes

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
        return jnp.asarray(self.predict_proba(S))

    def partial_fit(self, X, y):
        self.__setup(y)

        X_ = jnp.hstack((X, jnp.ones((len(X), 1))))

        target = label_binarize(y, classes=self.classes)

        if self.num_classes == 2:
            target = jnp.hstack([1 - target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * jnp.hstack([X_] * m)).reshape((n, m, m))

        logging.debug(f"{self.method=}")

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if self.optimizer in ["newton", "auto"] and self.num_classes <= 36:
            weights = _newton_update(
                self.weights_0_,
                X_,
                XXT,
                target,  # type: ignore
                self.num_classes,
                self.method,
                reg_lambda=self.reg_lambda,
                reg_mu=self.reg_mu,
                ref_row=self.ref_row,
                initializer=self.initializer,
                reg_format=self.reg_format,
            )

        elif self.optimizer in ["fmin_l_bfgs_b", "auto"] and self.num_classes > 36:

            def _gradient_np(*args, **kwargs):
                return np.array(_gradient(*args, **kwargs))

            res = scipy.optimize.fmin_l_bfgs_b(
                func=_objective,
                fprime=_gradient_np,
                x0=self.weights_0_,
                args=(
                    X_,
                    XXT,
                    target,
                    self.num_classes,
                    self.method,
                    self.reg_lambda,
                    self.reg_mu,
                    self.ref_row,
                    self.initializer,
                    self.reg_format,
                ),
                maxls=128,
                factr=1.0,
            )
            weights = res[0]
        else:
            raise (ValueError("Unknown optimizer: {}".format(self.optimizer)))

        self.weights_ = _get_weights(weights, self.num_classes, self.ref_row, self.method)

        return self

    def fit(self, X, y):
        self.__setup(y)

        X_ = jnp.hstack((X, jnp.ones((len(X), 1))))

        self.classes = np.unique(y)

        k = len(self.classes)

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (k * (k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (k * (k - 1))
                self.reg_mu = self.reg_mu / k

        target = label_binarize(y, classes=self.classes)

        if k == 2:
            target = jnp.hstack([1 - target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * jnp.hstack([X_] * m)).reshape((n, m, m))

        logging.debug(self.method)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if self.optimizer in ["newton", "auto"] and k <= 36:
            weights = _newton_update(
                self.weights_0_,
                X_,
                XXT,
                target,
                k,
                self.method,
                reg_lambda=self.reg_lambda,
                reg_mu=self.reg_mu,
                ref_row=self.ref_row,
                initializer=self.initializer,
                reg_format=self.reg_format,
            )
        elif self.optimizer in ["fmin_l_bfgs_b", "auto"] and k > 36:

            def _gradient_np(*args, **kwargs):
                return np.array(_gradient(*args, **kwargs))

            res = scipy.optimize.fmin_l_bfgs_b(
                func=_objective,
                fprime=_gradient_np,
                x0=self.weights_0_,
                args=(
                    X_,
                    XXT,
                    target,
                    k,
                    self.method,
                    self.reg_lambda,
                    self.reg_mu,
                    self.ref_row,
                    self.initializer,
                    self.reg_format,
                ),
                maxls=128,
                factr=1.0,
            )
            weights = res[0]
        else:
            raise (ValueError(f"Unknown optimizer: {self.optimizer}"))

        self.weights_ = _get_weights(weights, k, self.ref_row, self.method)

        return self

    def _get_initial_weights(self, ref_row, initializer: Literal["identity"] | None = "identity"):
        """
        Returns an array containing only the weights of the full weight
        matrix.
        """

        assert initializer in ["identity", None], "invalid initializer"

        if self.weights_0_ is not None:
            return self.weights_0_

        if initializer == "identity":
            return _get_identity_weights(self.num_classes, ref_row, self.method)
        else:
            if self.method == "Full":
                return jnp.zeros(self.num_classes * (self.num_classes + 1))
            elif self.method == "Diag":
                return jnp.zeros(2 * self.num_classes)
            elif self.method == "FixDiag":
                return jnp.zeros(1)


def _objective(
    params: jax.Array,
    X: jax.Array,
    XXT: jax.Array,
    y: jax.Array,
    k: int,
    method: Literal["Full", "Diag", "FixDiag"],
    reg_lambda: float,
    reg_mu: float | None,
    ref_row: bool,
    initializer: Literal["identity"] | None,
    reg_format: Literal["identity"] | None,
) -> jax.Array:
    weights = _get_weights(params, k, ref_row, method)

    outputs = clip_jax(_calculate_outputs(weights, X))
    loss = jnp.mean(-jnp.log(jnp.sum(y * outputs, axis=1)))

    if reg_mu is None:
        if reg_format == "identity":
            reg = jnp.hstack([jnp.eye(k), jnp.zeros((k, 1))])
        else:
            reg = jnp.zeros((k, k + 1))
        loss = loss + reg_lambda * jnp.sum((weights - reg) ** 2)
    else:
        weights_hat = weights - jnp.hstack([weights[:, :-1] * jnp.eye(k), jnp.zeros((k, 1))])
        loss = loss + reg_lambda * jnp.sum(weights_hat[:, :-1] ** 2) + reg_mu * jnp.sum(weights_hat[:, -1] ** 2)

    return loss


_gradient = jax.grad(_objective)


_hessian = jax.hessian(_objective)


def _get_weights(params, k: int, ref_row: bool, method: Literal["Full", "Diag", "FixDiag"]) -> jax.Array:
    """Reshapes the given params (weights) into the full matrix including 0"""

    if method in ["Full", None]:
        raw_weights = jnp.reshape(params, (-1, k + 1))

    elif method == "Diag":
        raw_weights = jnp.hstack([jnp.diag(params[:k]), jnp.reshape(params[k:], (-1, 1))])

    elif method == "FixDiag":
        raw_weights = jnp.hstack([jnp.eye(k) * params[0], jnp.zeros((k, 1))])

    else:
        raise ValueError(f"Unknown calibration method {method}")

    if ref_row:
        return raw_weights - jnp.repeat(jnp.reshape(raw_weights[-1, :], (1, -1)), k, axis=0)

    return raw_weights


def _get_identity_weights(k: int, ref_row: bool, method) -> jax.Array:
    if method in ["Full", None]:
        raw_weights = jnp.hstack([jnp.eye(k), jnp.zeros([k, 1])])

    elif method == "Diag":
        raw_weights = jnp.hstack([jnp.ones(k), jnp.zeros(k)])

    elif method == "FixDiag":
        raw_weights = jnp.ones(1)

    else:
        raise ValueError(f"Unknown calibration method {method}")

    # será q é necessário?
    # if ref_row:
    #     raw_weights = raw_weights - jnp.repeat(jnp.reshape(raw_weights[-1, :], [1, -1]), k, axis=0)

    return jnp.ravel(raw_weights)


def _calculate_outputs(weights: jax.Array, X: jax.Array) -> jax.Array:
    assert (
        weights.transpose().shape[0] == X.shape[1]
    ), f"{weights.transpose().shape[0]=} and {X.shape[1]=} should be equal"
    return _softmax(jnp.dot(X, weights.transpose()))


def _softmax(X: jax.Array) -> jax.Array:
    """Compute the softmax of matrix X in a numerically stable way."""
    X = X - jnp.max(X, axis=1, keepdims=True)
    exps = jnp.exp(X)

    return exps / jnp.sum(exps, axis=1, keepdims=True)


def _newton_update(
    weights_0: jax.Array,
    X: jax.Array,
    XX_T: jax.Array,
    target: jax.Array,
    k: int,
    method_: Literal["Full", "Diag", "FixDiag"],
    maxiter: int = 1024,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    reg_lambda: float = 0.0,
    reg_mu: float | None = None,
    ref_row: bool = True,
    initializer: Literal["identity"] | None = None,
    reg_format: Literal["identity"] | None = None,
) -> jax.Array:
    L_list = [
        float(
            _objective(
                weights_0,
                X,
                XX_T,
                target,
                k,
                method_,
                reg_lambda,
                reg_mu,
                ref_row,
                initializer,
                reg_format,
            )
        )
    ]

    weights = jnp.ravel(weights_0)

    # TODO move this to the initialization
    if method_ is None:
        weights = jnp.zeros_like(weights)

    last_idx, last_gradient = 0, 0
    newton_loop = tqdm(range(maxiter), position=2, leave=False, desc="_newton_update: ", postfix={"loss": 0.0})
    for idx in newton_loop:
        last_idx = idx

        gradient = _gradient(
            weights,
            X,
            XX_T,
            target,
            k,
            method_,
            reg_lambda,
            reg_mu,
            ref_row,
            initializer,
            reg_format,
        )

        if jnp.abs(gradient).sum() < gtol:
            logging.debug(f"{jnp.abs(gradient).sum()=} < {gtol=}")
            break

        last_gradient = gradient

        # FIXME hessian is ocasionally NaN
        hessian = _hessian(
            weights,
            X,
            XX_T,
            target,
            k,
            method_,
            reg_lambda,
            reg_mu,
            ref_row,
            initializer,
            reg_format,
        )

        if method_ == "FixDiag":
            updates = gradient / hessian
        else:
            try:
                inverse = scipy.linalg.pinv(hessian)
                updates = jnp.matmul(inverse, gradient)
            except (np.linalg.LinAlgError, ValueError) as err:
                logging.error(err)
                updates = gradient

        for step_size in jnp.hstack((jnp.linspace(1, 0.1, 10), jnp.logspace(-2, -32, 31))):
            logging.debug(f"step_size = {step_size}, updates = {updates.shape}, weights = {weights.shape}")
            logging.debug(f"mul = {(updates * step_size).shape}, jnp.ravel = {jnp.ravel(updates * step_size).shape}")
            temp_weights = weights - jnp.ravel(updates * step_size)

            if jnp.any(jnp.isnan(temp_weights)):
                logging.error(f"{method_}: There are NaNs in tmp_w")

            L = _objective(
                temp_weights,
                X,
                XX_T,
                target,
                k,
                method_,
                reg_lambda,
                reg_mu,
                ref_row,
                initializer,
                reg_format,
            )

            if (L - L_list[-1]) < 0:
                break

        L_list.append(float(L))

        logging.debug(
            f"{method_}: after {last_idx} iterations log-loss = {L:.7e}, sum_grad = {jnp.abs(gradient).sum():.7e}"
        )

        if jnp.isnan(L):
            logging.error(f"{method_}: log-loss is NaN")
            break

        if idx >= 5:
            if (float(np.min(np.diff(L_list[-5:]))) > -ftol) & (float(np.sum(np.diff(L_list[-5:])) > 0) == 0):
                weights = temp_weights.copy()
                logging.debug(f"{method_}: Terminate as there is not enough changes on loss.")
                break

        if (L_list[-1] - L_list[-2]) > 0:
            logging.debug(f"{method_}: Terminate as the loss increased {jnp.diff(L_list[-2:])}.")
            break
        else:
            weights = temp_weights.copy()

        newton_loop.set_postfix(loss=L)

    L = _objective(
        weights,  # type: ignore
        X,
        XX_T,
        target,
        k,
        method_,
        reg_lambda,
        reg_mu,
        ref_row,
        initializer,
        reg_format,
    )

    logging.debug(
        f"{method_}: after {last_idx} iterations final log-loss = {L:.7e}, sum_grad = {jnp.abs(last_gradient).sum():.7e}"
    )

    return weights
