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
        num_classes: int,
        weights: Any = None,
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
        if method not in ("Full", "Diag", "FixDiag"):
            raise ValueError(f"method {method} is invalid")

        self.num_classes = num_classes
        self.raw_weights = self.init_weights = weights
        self.method = method
        self.initializer = initializer
        self.reg_format = reg_format
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # If number, then ODIR is applied
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer

    def __post_init__(self):
        k = self.num_classes
        self.classes = np.array(range(k), dtype=np.uint8)
        self.raw_weights = self.__get_initial_weights(self.initializer)  # type: ignore Literal error

        if not self.reg_norm:
            return

        if self.reg_mu is None:
            self.reg_lambda = self.reg_lambda / (k * (k + 1))
            return

        self.reg_lambda = self.reg_lambda / (k * (k - 1))
        self.reg_mu = self.reg_mu / k

    def __get_initial_weights(self, initializer: Literal["identity"] | None = "identity"):
        """
        Returns an array containing only the weights of the full weight
        matrix.
        """

        assert initializer in ("identity", None), "invalid initializer"

        if self.weights is not None:
            return self.weights

        if initializer == "identity":
            return _get_identity_weights(self.num_classes, self.method)  # type: ignore

        match self.method:
            case "Full":
                return jnp.zeros(self.num_classes * (self.num_classes + 1))

            case "Diag":
                return jnp.zeros(2 * self.num_classes)

            case "FixDiag":
                return jnp.zeros(1)

    def partial_fit(self, X, y):
        k = self.num_classes
        X_ = jnp.hstack((X, jnp.ones((len(X), 1))))

        target = jnp.asarray(label_binarize(y, classes=self.classes))

        if k == 2:
            target = jnp.hstack([1 - target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * jnp.hstack([X_] * m)).reshape((n, m, m))

        if self.optimizer in ("newton", "auto") and k <= 36:
            weights = _newton_update(
                self.raw_weights,
                X_,
                XXT,
                target,
                k,
                self.method,  # type: ignore
                reg_lambda=self.reg_lambda,
                reg_mu=self.reg_mu,
                ref_row=self.ref_row,
                initializer=self.initializer,  # type: ignore
                reg_format=self.reg_format,  # type: ignore
            )

            self.raw_weights = weights
            self.weights = _get_weights(weights, k, self.ref_row, self.method)  # type: ignore

            return self

        if self.optimizer not in ("fmin_l_bfgs_b", "auto") or k <= 36:
            raise (ValueError("Unknown optimizer: {}".format(self.optimizer)))

        weights = scipy.optimize.fmin_l_bfgs_b(
            func=_objective,
            fprime=_gradient_np,
            x0=self.weights,
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
        )[0]

        self.raw_weights = weights
        self.weights = _get_weights(weights, k, self.ref_row, self.method)  # type: ignore

        return self

    def fit(self, X, y):
        X_ = jnp.hstack((X, jnp.ones((len(X), 1))))
        k = self.num_classes

        target = jnp.array(label_binarize(y, classes=self.classes))

        if k == 2:
            target = jnp.hstack([1 - target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * jnp.hstack([X_] * m)).reshape((n, m, m))

        if self.optimizer in ("newton", "auto") and k <= 36:
            weights = _newton_update(
                weights_0=self.weights,
                X=X_,
                XX_T=XXT,
                target=target,
                k=k,
                method_=self.method,  # type: ignore # Literal error
                reg_lambda=self.reg_lambda,
                reg_mu=self.reg_mu,
                ref_row=self.ref_row,
                initializer=self.initializer,  # type: ignore
                reg_format=self.reg_format,  # type: ignore
            )

            self.weights = _get_weights(weights, k, self.ref_row, self.method)  # type: ignore

            return self

        if self.optimizer not in ("fmin_l_bfgs_b", "auto") or k <= 36:
            raise (ValueError(f"Unknown optimizer: {self.optimizer}"))

        weights = scipy.optimize.fmin_l_bfgs_b(
            func=_objective,
            fprime=_gradient_np,
            x0=self.raw_weights,
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
        )[0]

        self.weights = _get_weights(weights, k, self.ref_row, self.method)  # type: ignore

        return self

    @property
    def coef_(self):
        return self.weights[:, :-1]

    @property
    def intercept_(self):
        return self.weights[:, -1]

    def predict_proba(self, S):
        S_ = jnp.hstack((S, jnp.ones((len(S), 1))))

        return jnp.asarray(_calculate_outputs(self.weights, S_))

    # FIXME Should we change predict for the argmax?
    def predict(self, S):
        return jnp.asarray(self.predict_proba(S))


def _gradient_np(*args, **kwargs):
    return np.array(_gradient(*args, **kwargs))


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

        return loss + reg_lambda * jnp.sum((weights - reg) ** 2)

    weights_hat = weights - jnp.hstack([weights[:, :-1] * jnp.eye(k), jnp.zeros((k, 1))])
    loss = loss + reg_lambda * jnp.sum(weights_hat[:, :-1] ** 2) + reg_mu * jnp.sum(weights_hat[:, -1] ** 2)

    return loss


_gradient = jax.grad(_objective)


_hessian = jax.hessian(_objective)


def _get_weights(
    raw_weights: Any, num_classes: int, ref_row: bool, method: Literal["Full", "Diag", "FixDiag"]
) -> jax.Array:
    """Reshapes the given params (weights) into the full matrix including 0"""

    match method:
        case "Full" | None:
            weights = jnp.reshape(raw_weights, (-1, num_classes + 1))

        case "Diag":
            weights = jnp.hstack(
                [jnp.diag(raw_weights[:num_classes]), jnp.reshape(raw_weights[num_classes:], (-1, 1))]
            )

        case "FixDiag":
            weights = jnp.hstack([jnp.eye(num_classes) * raw_weights[0], jnp.zeros((num_classes, 1))])

        case _:
            raise ValueError(f"Unknown calibration method {method}")

    if ref_row:
        return weights - jnp.repeat(jnp.reshape(weights[-1, :], (1, -1)), num_classes, axis=0)

    return weights


def _get_identity_weights(num_classes: int, method: Literal["Full", "Diag", "FixDiag"] | None) -> jax.Array:
    match method:
        case "Full" | None:
            return jnp.ravel(jnp.hstack([jnp.eye(num_classes), jnp.zeros([num_classes, 1])]))

        case "Diag":
            return jnp.ravel(jnp.hstack([jnp.ones(num_classes), jnp.zeros(num_classes)]))

        case "FixDiag":
            return jnp.ravel(jnp.ones(1))

    raise ValueError(f"Unknown calibration method {method}")


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
    loss = float(
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
    losses = [loss]

    weights: jax.Array = jnp.ravel(weights_0)

    # TODO move this to the initialization
    if method_ is None:
        weights = jnp.zeros_like(weights)

    last_idx, last_gradient = 0, 0

    newton_loop = tqdm(
        range(maxiter),
        position=2,
        leave=False,
        desc="_newton_update: ",
        postfix={"loss": 0.0},
    )
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

        if jnp.sum(jnp.abs(gradient)) < gtol:
            logging.debug(f"{jnp.sum(jnp.abs(gradient))=} < {gtol=}")
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
            updates: jax.Array = gradient / hessian
        else:
            try:
                inverse = scipy.linalg.pinv(hessian)
                updates: jax.Array = jnp.matmul(inverse, gradient)
            except (np.linalg.LinAlgError, ValueError) as err:
                logging.error(err)
                updates: jax.Array = gradient

        temp_weights = weights
        for step_size in jnp.hstack((jnp.linspace(1, 0.1, 10), jnp.logspace(-2, -32, 31))):
            logging.debug(f"step_size = {step_size}, updates = {updates.shape}, weights = {weights.shape}")
            logging.debug(f"mul = {(updates * step_size).shape}, jnp.ravel = {jnp.ravel(updates * step_size).shape}")

            temp_weights = weights - jnp.ravel(updates * step_size)

            if jnp.any(jnp.isnan(temp_weights)):
                logging.error(f"{method_}: There are NaNs in tmp_w")

            loss = _objective(
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

            if len(losses) > 5:  # keep losses small
                losses = losses[-5:]

            if jnp.isnan(loss):
                logging.error(f"{method_}: log-loss is NaN")
                break

            if loss < losses[-1]:  # find the first loss that is lesser than the last loss
                break

        logging.debug(
            f"{method_}: after {last_idx} iterations log-loss = {loss:.7e}, sum_grad = {jnp.sum(jnp.abs(gradient)):.7e}"
        )

        losses.append(float(loss))

        if losses[-1] > losses[-2]:
            logging.debug(f"{method_}: Terminate as the loss increased.")
            break

        weights = jnp.copy(temp_weights)
        newton_loop.set_postfix(loss=round(loss, 7))

        if idx >= 5 and np.min(np.diff(losses[-5:])) > -ftol:
            logging.debug(f"{method_}: Terminate as there is not enough changes on loss.")
            break

    loss = _objective(
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

    logging.debug(
        f"{method_}: after {last_idx} iterations final log-loss = {loss:.7e}, sum_grad = {jnp.sum(jnp.abs(last_gradient)):.7e}"
    )

    return weights
