"""
Partition Coloring Problem (PCP)

This package provides solvers for the Partition Coloring Problem.
"""

from .aco_solver import ACOResult, ACOSolver
from .dsatur import count_colors, dsatur_coloring, verify_coloring
from .ilp_solver import ILPSolver, SolverResult, SolverStatus
from .instance import PCPInstance

__all__ = [
    # Instance
    "PCPInstance",
    # ILP Solver
    "ILPSolver",
    "SolverResult",
    "SolverStatus",
    # ACO Solver
    "ACOSolver",
    "ACOResult",
    # DSATUR utilities
    "dsatur_coloring",
    "count_colors",
    "verify_coloring",
]
