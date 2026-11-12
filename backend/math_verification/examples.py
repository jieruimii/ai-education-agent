"""Reviewed deterministic examples available in desktop Mock Mode."""

from __future__ import annotations

from .math_sandbox import SandboxResult, execute_math_script


RLC_PROBLEM = (
    "A source-free series RLC circuit has R = 20 ohm, L = 0.500 henry, and "
    "C = 100 microfarad. The initial capacitor charge is q(0) = 10.0 millicoulomb "
    "and the initial current is i(0) = dq/dt = 0 ampere. Solve the second-order "
    "differential equation L*d2q/dt2 + R*dq/dt + q/C = 0 symbolically, determine "
    "whether the response is underdamped, critically damped, or overdamped, and "
    "report q at t = 25.0 millisecond in coulomb."
)


RLC_SCRIPT = r'''
import sympy as sp
import pint

t = sp.symbols("t", real=True, nonnegative=True)
q = sp.Function("q")
R_value = Q_(20.0, "ohm")
L_value = Q_(0.500, "henry")
C_value = Q_(100.0, "microfarad")
q0_value = Q_(10.0, "millicoulomb")
i0_value = Q_(0.0, "ampere")

representative_q = Q_(1.0, "coulomb")
representative_i = Q_(1.0, "ampere")
representative_didt = Q_(1.0, "ampere / second")
require_compatible(L_value * representative_didt, R_value * representative_i, "RLC voltage terms")
require_compatible(R_value * representative_i, representative_q / C_value, "RLC voltage terms")
require_compatible(R_value**2, L_value / C_value, "RLC damping discriminant")
require_dimension(q0_value, "coulomb", "initial charge")
require_dimension(i0_value, "ampere", "initial current")

R = sp.Symbol("R", positive=True)
L = sp.Symbol("L", positive=True)
C = sp.Symbol("C", positive=True)
equation = sp.Eq(L * sp.diff(q(t), t, 2) + R * sp.diff(q(t), t) + q(t) / C, 0)
symbolic_general = sp.dsolve(equation, q(t))
numeric_equation = equation.subs({
    R: R_value.to("ohm").magnitude,
    L: L_value.to("henry").magnitude,
    C: C_value.to("farad").magnitude,
})
solution = sp.dsolve(
    numeric_equation,
    q(t),
    ics={
        q(0): q0_value.to("coulomb").magnitude,
        sp.diff(q(t), t).subs(t, 0): i0_value.to("ampere").magnitude,
    },
)
discriminant = (
    R_value.to("ohm").magnitude**2
    - 4 * L_value.to("henry").magnitude / C_value.to("farad").magnitude
)
classification = "underdamped" if discriminant < 0 else "critically damped" if discriminant == 0 else "overdamped"
q_at_time = solution.rhs.subs(t, Q_(25.0, "millisecond").to("second").magnitude).evalf()
emit_result(
    symbolic_solution=solution,
    numerical_value=Q_(float(sp.re(q_at_time)), "coulomb"),
    target_units="coulomb",
    verification=[classification, str(symbolic_general)],
)
'''


def solve_reviewed_rlc_example(significant_figures: int = 4) -> SandboxResult:
    return execute_math_script(
        RLC_SCRIPT,
        requires_units=True,
        angle_mode="unspecified",
        significant_figures=significant_figures,
    )


def render_reviewed_rlc_answer(result: SandboxResult) -> str:
    if result.status != "success":
        return (
            "The reviewed RLC example could not be executed safely. "
            f"Sandbox status: {result.status}. {result.error_message or ''}"
        ).strip()
    classification = next(
        (item for item in result.verification if item in {
            "underdamped", "critically damped", "overdamped"
        }),
        "classification unavailable",
    )
    checks = "\n".join(f"- {item}" for item in result.verification[:-1])
    return (
        "The verified circuit response is underdamped, and the deterministic solver gives "
        f"q(25.0 ms) = {result.numerical_result} {result.units}.\n\n"
        "## Governing equation\n"
        "The source-free series circuit obeys $Lq'' + Rq' + q/C = 0$.\n\n"
        "## Symbolic solution\n"
        f"`{result.symbolic_solution}`\n\n"
        "## Deterministic verification\n"
        f"- Damping classification: {classification}\n{checks}\n"
        "- SymPy solved the differential equation; Pint checked every unit-bearing relation.\n\n"
        "## Final result\n"
        f"$q(25.0\\,\\mathrm{{ms}}) = {result.numerical_result}\\,\\mathrm{{{result.units}}}$"
    )
