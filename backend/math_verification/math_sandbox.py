"""Deterministic SymPy/Pint execution with an allowlisted Python subset.

This module is deliberately usable both as a library and as the isolated worker
process launched by :func:`execute_math_script`.  It is defense in depth, not a
replacement for a container boundary.  In production, run the FastAPI service
inside an unprivileged container with a read-only filesystem and no network.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


MAX_SCRIPT_BYTES = 32_000
MAX_OUTPUT_BYTES = 32_000
RESULT_MARKER = "__MATH_SANDBOX_RESULT__="


class SandboxResult(BaseModel):
    status: Literal[
        "success", "validation_error", "unit_error", "timeout", "execution_error"
    ]
    symbolic_solution: str | None = None
    numerical_result: str | None = None
    units: str | None = None
    error_message: str | None = None
    verification: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    execution_id: str


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 8.0
    cpu_seconds: int = 6
    memory_bytes: int = 512 * 1024 * 1024
    file_bytes: int = 0
    open_files: int = 32


class ScriptValidationError(ValueError):
    """Raised before execution when generated code leaves the safe subset."""


ALLOWED_IMPORTS = {"sympy", "pint", "math"}
ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

FORBIDDEN_NAMES = {
    "breakpoint", "compile", "delattr", "eval", "exec", "exit", "getattr",
    "globals", "help", "input", "locals", "memoryview", "open", "quit",
    "setattr", "type", "vars", "__import__",
}
FORBIDDEN_ATTRIBUTES = {
    "autowrap", "codegen", "evalf_subs", "exec", "frompyfunc", "import_module",
    "lambdify", "load", "loads", "mro", "open", "read", "read_text",
    "read_bytes", "run", "system", "ufuncify", "write", "write_text", "write_bytes",
}
SAFE_SYMPY_ATTRIBUTES = {
    "Abs", "Derivative", "Eq", "Function", "I", "Integral", "Matrix", "N",
    "Rational", "Symbol", "acos", "asin", "atan", "atan2", "cos", "diff",
    "discriminant", "dsolve", "exp", "factor", "im", "integrate", "limit",
    "linsolve", "log", "nonlinsolve", "pi", "re", "roots", "simplify", "sin",
    "solve", "solveset", "sqrt", "summation", "symbols", "tan",
}
SAFE_PINT_ATTRIBUTES = {"Quantity", "UnitRegistry"}
SAFE_MATH_ATTRIBUTES = {
    "acos", "asin", "atan", "atan2", "cos", "degrees", "e", "exp", "log",
    "pi", "radians", "sin", "sqrt", "tan",
}
ALLOWED_NODE_TYPES = {
    ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.Assign, ast.AnnAssign,
    ast.AugAssign, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Tuple, ast.List, ast.Dict, ast.Set, ast.keyword, ast.Attribute, ast.Subscript,
    ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If, ast.IfExp,
    ast.For, ast.comprehension, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
}
SYMBOLIC_CALLS = {
    "solve", "solveset", "linsolve", "nonlinsolve", "dsolve", "integrate",
    "diff", "limit", "summation", "Matrix", "Eq", "simplify", "factor",
}
TRIG_CALLS = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2"}
UNIT_CHECK_CALLS = {"require_compatible", "require_dimension"}


class SafeMathValidator(ast.NodeVisitor):
    def __init__(self, *, requires_units: bool, angle_mode: str) -> None:
        self.requires_units = requires_units
        self.angle_mode = angle_mode
        self.symbolic_call_seen = False
        self.unit_check_seen = False
        self.trig_call_seen = False
        self.angle_conversion_seen = False
        self.emit_calls = 0
        self.symbolic_call_lines: list[int] = []
        self.unit_check_lines: list[int] = []
        self.trig_call_lines: list[int] = []
        self.angle_conversion_lines: list[int] = []

    def validate(self, source: str) -> None:
        if not source.strip():
            raise ScriptValidationError("Script is empty.")
        if len(source.encode("utf-8")) > MAX_SCRIPT_BYTES:
            raise ScriptValidationError(f"Script exceeds {MAX_SCRIPT_BYTES} bytes.")
        if RESULT_MARKER in source:
            raise ScriptValidationError("Script contains a reserved sandbox marker.")
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise ScriptValidationError(f"Syntax error: {exc.msg} at line {exc.lineno}.") from exc
        if sum(1 for _ in ast.walk(tree)) > 2_000:
            raise ScriptValidationError("Script is too complex (maximum 2,000 AST nodes).")
        self.visit(tree)
        if self.emit_calls != 1:
            raise ScriptValidationError("Script must call emit_result exactly once.")
        if not self.symbolic_call_seen:
            raise ScriptValidationError(
                "A SymPy symbolic operation such as solve, dsolve, integrate, diff, or Matrix is required."
            )
        if self.requires_units and not self.unit_check_seen:
            raise ScriptValidationError(
                "Unit-bearing problems must call require_compatible or require_dimension before emit_result."
            )
        if (
            self.requires_units
            and self.symbolic_call_lines
            and min(self.unit_check_lines) > min(self.symbolic_call_lines)
        ):
            raise ScriptValidationError(
                "Dimensional checks must occur before the first symbolic solve/calculus operation."
            )
        if self.trig_call_seen and self.angle_mode == "unspecified":
            raise ScriptValidationError("Trigonometry requires angle_mode='degrees' or 'radians'.")
        if self.trig_call_seen and not self.angle_conversion_seen:
            raise ScriptValidationError(
                "Trigonometric code must call angle_to_radians explicitly."
            )
        if (
            self.trig_call_lines
            and self.angle_conversion_lines
            and min(self.angle_conversion_lines) > min(self.trig_call_lines)
        ):
            raise ScriptValidationError(
                "angle_to_radians must be called before the first trigonometric operation."
            )

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODE_TYPES:
            raise ScriptValidationError(f"Disallowed syntax: {type(node).__name__}.")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                raise ScriptValidationError(f"Import not allowed: {alias.name}.")
            expected_alias = {"sympy": "sp", "pint": None, "math": None}[alias.name]
            if alias.asname != expected_alias:
                required = f" as {expected_alias}" if expected_alias else " without an alias"
                raise ScriptValidationError(f"Import {alias.name}{required} is required.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ScriptValidationError("Use module-qualified imports only (import sympy as sp; import pint).")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
            raise ScriptValidationError(f"Name not allowed: {node.id}.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
            raise ScriptValidationError(f"Attribute not allowed: {node.attr}.")
        if isinstance(node.value, ast.Name):
            module_rules = {
                "sp": SAFE_SYMPY_ATTRIBUTES,
                "sympy": SAFE_SYMPY_ATTRIBUTES,
                "pint": SAFE_PINT_ATTRIBUTES,
                "math": SAFE_MATH_ATTRIBUTES,
            }
            allowed = module_rules.get(node.value.id)
            if allowed is not None and node.attr not in allowed:
                raise ScriptValidationError(
                    f"Module attribute not allowed: {node.value.id}.{node.attr}."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in FORBIDDEN_NAMES or leaf in FORBIDDEN_ATTRIBUTES:
            raise ScriptValidationError(f"Call not allowed: {name}.")
        if leaf in SYMBOLIC_CALLS:
            self.symbolic_call_seen = True
            self.symbolic_call_lines.append(node.lineno)
        if leaf in UNIT_CHECK_CALLS:
            self.unit_check_seen = True
            self.unit_check_lines.append(node.lineno)
        if leaf in TRIG_CALLS:
            self.trig_call_seen = True
            self.trig_call_lines.append(node.lineno)
        if leaf == "angle_to_radians":
            self.angle_conversion_seen = True
            self.angle_conversion_lines.append(node.lineno)
        if leaf == "emit_result":
            self.emit_calls += 1
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def validate_script(
    script: str, *, requires_units: bool, angle_mode: Literal["degrees", "radians", "unspecified"]
) -> None:
    SafeMathValidator(requires_units=requires_units, angle_mode=angle_mode).validate(script)


def _preexec_limits(limits: SandboxLimits):
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (OSError, ValueError):
            # Some limits are unavailable on macOS. The production container must
            # provide the hard boundary; timeout and AST checks still apply here.
            pass

    return apply


def _truncate(value: str | bytes | None, limit: int = MAX_OUTPUT_BYTES) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="replace") + "\n...[truncated]"


def execute_math_script(
    script: str,
    *,
    requires_units: bool,
    angle_mode: Literal["degrees", "radians", "unspecified"],
    significant_figures: int = 4,
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Validate and execute one generated math script in a child process."""

    execution_id = uuid.uuid4().hex
    limits = limits or SandboxLimits()
    if not 2 <= significant_figures <= 8:
        return SandboxResult(
            status="validation_error",
            error_message="significant_figures must be between 2 and 8.",
            execution_id=execution_id,
        )
    try:
        validate_script(script, requires_units=requires_units, angle_mode=angle_mode)
    except ScriptValidationError as exc:
        return SandboxResult(
            status="validation_error", error_message=str(exc), execution_id=execution_id
        )

    payload = json.dumps(
        {
            "script": script,
            "requires_units": requires_units,
            "angle_mode": angle_mode,
            "significant_figures": significant_figures,
            "execution_id": execution_id,
        }
    )
    with tempfile.TemporaryDirectory(prefix="math-sandbox-") as temp_dir:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": temp_dir,
            "TMPDIR": temp_dir,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--math-sandbox-worker"]
        else:
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"]
        try:
            completed = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=limits.timeout_seconds,
                cwd=temp_dir,
                env=env,
                check=False,
                preexec_fn=_preexec_limits(limits) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                status="timeout",
                error_message=f"Execution exceeded {limits.timeout_seconds:.1f} seconds.",
                stdout=_truncate(exc.stdout or ""),
                stderr=_truncate(exc.stderr or ""),
                execution_id=execution_id,
            )

    stdout = _truncate(completed.stdout)
    stderr = _truncate(completed.stderr)
    marker_position = stdout.rfind(RESULT_MARKER)
    if marker_position >= 0:
        result_json = stdout[marker_position + len(RESULT_MARKER):].splitlines()[0]
        visible_stdout = stdout[:marker_position].rstrip("\n")
        try:
            result = SandboxResult.model_validate_json(result_json)
            result.stdout = _truncate(visible_stdout)
            result.stderr = stderr
            return result
        except Exception as exc:  # pragma: no cover - worker corruption path
            return SandboxResult(
                status="execution_error",
                error_message=f"Worker returned invalid structured output: {type(exc).__name__}.",
                stdout=visible_stdout,
                stderr=stderr,
                execution_id=execution_id,
            )
    visible_stdout = stdout
    return SandboxResult(
        status="execution_error",
        error_message=f"Worker exited with code {completed.returncode} without a result.",
        stdout=visible_stdout,
        stderr=stderr,
        execution_id=execution_id,
    )


def _engineering_number(value: Any, significant_figures: int) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Final numerical value must be finite.")
    if numeric == 0:
        return "0"
    exponent = (
        0
        if 1e-3 <= abs(numeric) < 1e3
        else int(math.floor(math.log10(abs(numeric)) / 3.0) * 3)
    )
    mantissa = numeric / (10**exponent)
    decimals = max(0, significant_figures - 1 - int(math.floor(math.log10(abs(mantissa)))))
    rounded = f"{mantissa:.{decimals}f}"
    return rounded if exponent == 0 else f"{rounded}e{exponent:+d}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _run_worker(payload: dict[str, Any]) -> SandboxResult:
    import pint
    import sympy as sp
    from pint.errors import DimensionalityError

    execution_id = str(payload["execution_id"])
    requires_units = bool(payload["requires_units"])
    angle_mode = str(payload["angle_mode"])
    significant_figures = int(payload["significant_figures"])
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    checks: list[str] = []
    emitted: dict[str, Any] = {}

    def require_compatible(left: Any, right: Any, context: str = "equation") -> bool:
        try:
            left_dim = left.dimensionality
            right_dim = right.dimensionality
        except AttributeError as exc:
            raise TypeError("require_compatible expects two Pint quantities.") from exc
        if left_dim != right_dim:
            raise DimensionalityError(left.units, right.units, extra_msg=f" in {context}")
        checks.append(f"Compatible dimensions verified for {context}: {left_dim}.")
        return True

    def require_dimension(quantity: Any, expected: str, name: str = "quantity") -> bool:
        expected_quantity = 1 * ureg.parse_units(expected)
        if quantity.dimensionality != expected_quantity.dimensionality:
            raise DimensionalityError(
                quantity.units, expected_quantity.units, extra_msg=f" for {name}"
            )
        checks.append(f"Dimension verified for {name}: {quantity.dimensionality}.")
        return True

    def angle_to_radians(value: Any, mode: str | None = None) -> float:
        selected = mode or angle_mode
        if selected not in {"degrees", "radians"}:
            raise ValueError("Angle mode must be explicitly 'degrees' or 'radians'.")
        if hasattr(value, "to"):
            radians = float(value.to(ureg.radian).magnitude)
        elif selected == "degrees":
            radians = math.radians(float(value))
        else:
            radians = float(value)
        checks.append(f"Angle explicitly converted from {selected} to radians.")
        return radians

    def emit_result(
        *,
        symbolic_solution: Any,
        numerical_value: Any,
        target_units: str | None = None,
        verification: list[str] | None = None,
    ) -> None:
        nonlocal emitted
        if emitted:
            raise RuntimeError("emit_result may only be called once.")
        if requires_units and not checks:
            raise RuntimeError("No dimensional verification was completed.")
        units: str | None = None
        value = numerical_value
        if hasattr(numerical_value, "to") and hasattr(numerical_value, "magnitude"):
            if target_units:
                value = numerical_value.to(ureg.parse_units(target_units))
            units = f"{value.units:~P}"
            magnitude = value.magnitude
        else:
            if requires_units:
                raise TypeError("Unit-bearing problems must emit a Pint Quantity.")
            if target_units:
                raise TypeError("target_units cannot be used with a dimensionless result.")
            magnitude = value
        if isinstance(magnitude, sp.Basic):
            magnitude = magnitude.evalf(significant_figures + 4)
        emitted = {
            "symbolic_solution": str(symbolic_solution),
            "numerical_result": _engineering_number(magnitude, significant_figures),
            "units": units,
            "verification": [*checks, *(_json_safe(verification or []))],
        }

    def safe_import(
        name: str,
        _globals: Any = None,
        _locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0 or fromlist or name not in ALLOWED_IMPORTS:
            raise ImportError(f"Import not allowed: {name}.")
        return {"sympy": sp, "pint": pint, "math": math}[name]

    safe_builtins = dict(ALLOWED_BUILTINS)
    safe_builtins["__import__"] = safe_import

    safe_globals = {
        "__builtins__": safe_builtins,
        "sp": sp,
        "sympy": sp,
        "pint": pint,
        "math": math,
        "ureg": ureg,
        "Q_": ureg.Quantity,
        "require_compatible": require_compatible,
        "require_dimension": require_dimension,
        "angle_to_radians": angle_to_radians,
        "emit_result": emit_result,
    }
    try:
        compiled = compile(payload["script"], "<generated-math>", "exec")
        exec(compiled, safe_globals, safe_globals)
        if not emitted:
            raise RuntimeError("Script completed without emit_result.")
        return SandboxResult(
            status="success", execution_id=execution_id, **emitted
        )
    except DimensionalityError as exc:
        return SandboxResult(
            status="unit_error",
            error_message=_truncate(f"{type(exc).__name__}: {exc}"),
            verification=checks,
            execution_id=execution_id,
        )
    except Exception:
        return SandboxResult(
            status="execution_error",
            error_message=_truncate(traceback.format_exc(limit=8)),
            verification=checks,
            execution_id=execution_id,
        )


def _worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        result = _run_worker(payload)
    except Exception:
        result = SandboxResult(
            status="execution_error",
            error_message=_truncate(traceback.format_exc(limit=8)),
            execution_id="worker-initialization",
        )
    # Windowed macOS builds may replace sys.stdout with None. Descriptor 1 is
    # still the pipe created by subprocess.run, so write the protocol directly.
    os.write(1, (RESULT_MARKER + result.model_dump_json() + "\n").encode("utf-8"))
    return 0 if result.status == "success" else 1


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    raise SystemExit(_worker_main())
