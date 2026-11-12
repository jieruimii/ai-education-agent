from __future__ import annotations

import unittest

from backend.math_verification.math_sandbox import SandboxLimits, execute_math_script
from backend.math_verification.examples import RLC_SCRIPT


FAST_LIMITS = SandboxLimits(timeout_seconds=12, cpu_seconds=10)


class MathSandboxTests(unittest.TestCase):
    def test_second_order_rlc_solution_is_verified(self) -> None:
        result = execute_math_script(
            RLC_SCRIPT,
            requires_units=True,
            angle_mode="unspecified",
            significant_figures=4,
            limits=FAST_LIMITS,
        )
        self.assertEqual(result.status, "success", result.error_message)
        self.assertEqual(result.units, "C")
        self.assertIn("q(t)", result.symbolic_solution or "")
        self.assertGreaterEqual(len(result.verification), 6)
        self.assertIn("underdamped", result.verification)

    def test_dimensional_mismatch_is_caught_before_solve(self) -> None:
        script = '''
import sympy as sp
import pint
x = sp.symbols("x")
force = Q_(1, "newton")
pressure = Q_(1, "pascal")
require_compatible(force, pressure, "invalid addition")
solution = sp.solve(sp.Eq(x + 1, 2), x)[0]
emit_result(symbolic_solution=solution, numerical_value=Q_(solution, "newton"), target_units="newton")
'''
        result = execute_math_script(
            script,
            requires_units=True,
            angle_mode="unspecified",
            limits=FAST_LIMITS,
        )
        self.assertEqual(result.status, "unit_error")
        self.assertIn("DimensionalityError", result.error_message or "")

    def test_unit_check_after_solve_is_rejected(self) -> None:
        script = '''
import sympy as sp
import pint
x = sp.symbols("x")
solution = sp.solve(sp.Eq(x + 1, 2), x)[0]
force = Q_(1, "newton")
require_compatible(force, force, "too late")
emit_result(symbolic_solution=solution, numerical_value=Q_(solution, "newton"), target_units="newton")
'''
        result = execute_math_script(
            script, requires_units=True, angle_mode="unspecified", limits=FAST_LIMITS
        )
        self.assertEqual(result.status, "validation_error")
        self.assertIn("before the first symbolic", result.error_message or "")

    def test_unsafe_import_is_rejected_without_execution(self) -> None:
        script = '''
import os
import sympy as sp
x = sp.symbols("x")
solution = sp.solve(sp.Eq(x, 1), x)[0]
emit_result(symbolic_solution=solution, numerical_value=solution)
'''
        result = execute_math_script(
            script, requires_units=False, angle_mode="unspecified", limits=FAST_LIMITS
        )
        self.assertEqual(result.status, "validation_error")
        self.assertIn("Import not allowed", result.error_message or "")

    def test_trigonometry_requires_explicit_angle_conversion(self) -> None:
        script = '''
import sympy as sp
theta = sp.symbols("theta")
solution = sp.solve(sp.Eq(theta, sp.sin(30)), theta)[0]
emit_result(symbolic_solution=solution, numerical_value=solution)
'''
        result = execute_math_script(
            script, requires_units=False, angle_mode="degrees", limits=FAST_LIMITS
        )
        self.assertEqual(result.status, "validation_error")
        self.assertIn("angle_to_radians", result.error_message or "")

    def test_degree_conversion_succeeds(self) -> None:
        script = '''
import sympy as sp
x = sp.symbols("x")
theta = angle_to_radians(30, "degrees")
solution = sp.solve(sp.Eq(x, sp.sin(theta)), x)[0]
emit_result(symbolic_solution=solution, numerical_value=solution)
'''
        result = execute_math_script(
            script,
            requires_units=False,
            angle_mode="degrees",
            significant_figures=4,
            limits=FAST_LIMITS,
        )
        self.assertEqual(result.status, "success", result.error_message)
        self.assertEqual(result.numerical_result, "0.5000")


if __name__ == "__main__":
    unittest.main()
