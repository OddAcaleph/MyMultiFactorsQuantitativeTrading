"""QP solver wrapper using CVXPY + OSQP."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import cvxpy as cp
import numpy as np

from optimizer.objective_builder import QPObjective
from optimizer.constraint_builder import QPConstraints


@dataclass
class QPSolution:
    """QP solution."""

    x: np.ndarray
    status: str
    objective_value: float
    solve_time_ms: float


class QPSolver:
    """Solve QP problems via CVXPY with OSQP backend.

    Using CVXPY as a modeling layer makes it easy to switch solvers and
    avoids manual matrix construction errors.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.solver_name: str = config.get("solver", "OSQP")
        self.max_iter: int = int(config.get("max_iter", 4000))
        self.eps_abs: float = float(config.get("eps_abs", 1e-6))
        self.eps_rel: float = float(config.get("eps_rel", 1e-6))
        self.warm_start: bool = bool(config.get("warm_start", True))
        self.verbose: bool = bool(config.get("verbose", False))

    def solve(self, objective: QPObjective, constraints: QPConstraints) -> QPSolution:
        """Solve a QP problem.

        Parameters
        ----------
        objective : QPObjective
            P, q for 0.5 x^T P x + q^T x.
        constraints : QPConstraints
            G, h, A, b, lb, ub.

        Returns
        -------
        QPSolution
        """
        n = objective.n_variables
        x = cp.Variable(n)

        # Objective: 0.5 * x^T P x + q^T x
        P_symmetric = (objective.P + objective.P.T) / 2.0
        objective_expr = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P_symmetric)) + objective.q @ x)

        # Constraints
        cons_list = []
        if constraints.G.shape[0] > 0:
            cons_list.append(constraints.G @ x <= constraints.h)
        if constraints.A.shape[0] > 0:
            cons_list.append(constraints.A @ x == constraints.b)
        if np.any(np.isfinite(constraints.lb)):
            cons_list.append(x >= constraints.lb)
        if np.any(np.isfinite(constraints.ub)):
            cons_list.append(x <= constraints.ub)

        # Quadratic / SOCP constraints
        if constraints.soc_constraints:
            for soc in constraints.soc_constraints:
                if soc.get("type") == "quad_form_le":
                    N = soc["n_weights"]
                    P = soc["P"]
                    rhs = soc["rhs"]
                    w = x[:N]
                    P_symmetric = (P + P.T) / 2.0
                    cons_list.append(cp.quad_form(w, cp.psd_wrap(P_symmetric)) <= rhs)

        # Choose solver: OSQP for pure QP (fast), CLARABEL for problems with
        # SOCP/quadratic constraints (OSQP doesn't support SOCP).
        has_soc = constraints.soc_constraints is not None and len(constraints.soc_constraints) > 0
        solver_name = self.solver_name
        if has_soc and solver_name.upper() == "OSQP":
            solver_name = "CLARABEL"

        prob = cp.Problem(objective_expr, cons_list)

        t0 = time.perf_counter()
        try:
            solver_kwargs = {
                "solver": solver_name,
                "max_iter": self.max_iter,
                "warm_start": self.warm_start,
                "verbose": self.verbose,
            }
            if solver_name.upper() == "OSQP":
                solver_kwargs["eps_abs"] = self.eps_abs
                solver_kwargs["eps_rel"] = self.eps_rel
            elif solver_name.upper() == "CLARABEL":
                solver_kwargs["tol_gap_abs"] = self.eps_abs
                solver_kwargs["tol_gap_rel"] = self.eps_rel

            prob.solve(**solver_kwargs)
        except cp.error.SolverError as e:
            return QPSolution(
                x=np.zeros(n),
                status=f"solver_error: {e}",
                objective_value=float("inf"),
                solve_time_ms=(time.perf_counter() - t0) * 1000,
            )

        solve_ms = (time.perf_counter() - t0) * 1000
        status = prob.status or "unknown"
        x_val = x.value if x.value is not None else np.zeros(n)
        obj_val = prob.value if prob.value is not None else float("inf")

        return QPSolution(
            x=x_val,
            status=status,
            objective_value=float(obj_val),
            solve_time_ms=float(solve_ms),
        )
