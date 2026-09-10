"""QP objective function builder.

Builds the standard QP form:

    minimize  (1/2) x^T P x + q^T x

where x = [w; t] and t is the turnover auxiliary variable (L1 penalty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QPObjective:
    """Quadratic objective: 0.5 * x^T P x + q^T x."""

    P: np.ndarray
    q: np.ndarray
    n_variables: int


class ObjectiveBuilder:
    """Build QP objective for mean-variance optimization with turnover penalty.

    Objective (maximization form):
        alpha^T w - lambda * w^T Σ w - gamma * ||w - w_prev||_1

    Converted to minimization:
        lambda * w^T Σ w + gamma * sum(t_i) - alpha^T w

    where t_i >= |w_i - w_prev_i| (handled by constraints).
    """

    def build(
        self,
        alpha: np.ndarray,
        exposures: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray,
        risk_aversion: float,
        turnover_penalty: float,
        prev_weights: np.ndarray,
    ) -> QPObjective:
        """Build the QP objective.

        Parameters
        ----------
        alpha : np.ndarray (N,)
            Alpha scores.
        exposures : np.ndarray (N, K)
            Factor exposure matrix.
        factor_cov : np.ndarray (K, K)
            Factor covariance matrix.
        specific_var : np.ndarray (N,)
            Specific (idiosyncratic) variance.
        risk_aversion : float
            Risk aversion coefficient lambda.
        turnover_penalty : float
            Turnover penalty coefficient gamma.
        prev_weights : np.ndarray (N,)
            Previous period weights.

        Returns
        -------
        QPObjective
            P and q for the extended variable x = [w; t].
        """
        N = len(alpha)

        # Sigma = X F X^T + diag(D)
        # We build the N×N quadratic matrix for the risk term.
        # For N up to a few hundred this is fine.
        sigma = exposures @ factor_cov @ exposures.T
        np.fill_diagonal(sigma, sigma.diagonal() + specific_var)

        # P matrix (2N x 2N): only the top-left N×N block is non-zero
        # P = 2 * lambda * Sigma  (because 0.5 x^T P x gives lambda w^T Σ w)
        P = np.zeros((2 * N, 2 * N), dtype=np.float64)
        P[:N, :N] = 2.0 * risk_aversion * sigma

        # q vector (2N,): [-alpha; gamma]
        q = np.zeros(2 * N, dtype=np.float64)
        q[:N] = -alpha
        q[N:] = turnover_penalty

        return QPObjective(P=P, q=q, n_variables=2 * N)
