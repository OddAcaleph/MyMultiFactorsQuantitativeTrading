"""Covariance matrix validator.

Checks that covariance matrices are symmetric, positive semi-definite,
free of NaN/Inf, and have reasonable condition numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class CovarianceValidationResult:
    """Result of covariance validation."""

    n_matrices: int = 0
    all_symmetric: bool = True
    all_psd: bool = True
    all_finite: bool = True
    all_positive_diag: bool = True
    min_eigenvalue_min: float = float("inf")
    min_eigenvalue_mean: float = 0.0
    condition_number_max: float = 0.0
    condition_number_mean: float = 0.0
    passed: bool = True
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_matrices": self.n_matrices,
            "all_symmetric": self.all_symmetric,
            "all_psd": self.all_psd,
            "all_finite": self.all_finite,
            "all_positive_diag": self.all_positive_diag,
            "min_eigenvalue_min": self.min_eigenvalue_min,
            "min_eigenvalue_mean": self.min_eigenvalue_mean,
            "condition_number_max": self.condition_number_max,
            "condition_number_mean": self.condition_number_mean,
            "passed": self.passed,
            "issues": self.issues,
        }


class CovarianceValidator:
    """Validate covariance matrices.

    Parameters
    ----------
    config
        Risk model config dictionary.
    eig_tol
        Tolerance for negative eigenvalues (default: -1e-8).
    cond_max
        Maximum acceptable condition number (default: 1e15).
        Factor model covariance matrices (XFX^T+D) naturally have high
        condition numbers because the systematic component has rank K << N,
        so the default is very permissive.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        eig_tol: float = -1e-8,
        cond_max: float = 1e15,
    ) -> None:
        self.config = dict(config or {})
        self.eig_tol = eig_tol
        self.cond_max = cond_max

    def validate(self, matrices: dict[int, np.ndarray]) -> CovarianceValidationResult:
        """Validate a set of covariance matrices.

        Parameters
        ----------
        matrices
            Mapping from trade_date to covariance matrix.
        """

        result = CovarianceValidationResult()
        result.n_matrices = len(matrices)

        if not matrices:
            result.passed = False
            result.issues.append("No covariance matrices to validate")
            return result

        min_eigs = []
        cond_nums = []

        for td, mat in matrices.items():
            if not np.all(np.isfinite(mat)):
                result.all_finite = False
                result.passed = False
                result.issues.append(f"Date {td}: matrix contains NaN or Inf")
                continue

            if not np.allclose(mat, mat.T, atol=1e-10):
                result.all_symmetric = False
                result.passed = False
                result.issues.append(f"Date {td}: matrix is not symmetric")
                continue

            diag = np.diag(mat)
            if np.any(diag <= 0):
                result.all_positive_diag = False
                result.passed = False
                result.issues.append(f"Date {td}: non-positive diagonal element(s)")
                continue

            eigvals = np.linalg.eigvalsh(mat)
            min_eig = float(eigvals.min())
            max_eig = float(eigvals.max())
            cond = max_eig / max(min_eig, 1e-12)

            min_eigs.append(min_eig)
            cond_nums.append(cond)

            if min_eig < self.eig_tol:
                result.all_psd = False
                result.passed = False
                result.issues.append(
                    f"Date {td}: min eigenvalue {min_eig:.6e} < tolerance {self.eig_tol}"
                )

            if cond > self.cond_max:
                result.passed = False
                result.issues.append(
                    f"Date {td}: condition number {cond:.2e} > {self.cond_max:.0e}"
                )

        if min_eigs:
            result.min_eigenvalue_min = float(min(min_eigs))
            result.min_eigenvalue_mean = float(np.mean(min_eigs))
            result.condition_number_max = float(max(cond_nums))
            result.condition_number_mean = float(np.mean(cond_nums))

        return result
