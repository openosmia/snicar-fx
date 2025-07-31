"""Radiative transfer solvers for snicar-fx.

This module contains implementations of the radiative transfer solver:
- Adding-doubling solver
"""

from snicarfx.rt_solvers.adding_doubling_solver import solve_adding_doubling

# restrict import star to solve method
__all__ = ["solve_adding_doubling"]
