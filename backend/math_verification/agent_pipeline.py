"""Provider-neutral tool loop for deterministic engineering mathematics."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from .math_sandbox import SandboxLimits, SandboxResult, execute_math_script


SYSTEM_PROMPT = """You are the parsing and explanation layer for a deterministic engineering math engine.

NON-NEGOTIABLE RULES
1. Never perform arithmetic, algebra, calculus, differential equations, matrix operations,
   trigonometry, or unit conversion in prose. All such work must be executed by the
   execute_verified_math tool using SymPy and Pint.
2. Your first response must contain exactly one execute_verified_math call and no proposed
   numerical answer. Put knowns, unknowns, governing equations, assumptions, and units in the
   tool arguments.
3. The script must solve symbolically before numerical substitution. Do not hard-code a final
   answer. Use import sympy as sp and import pint; ureg and Q_ are already supplied.
4. For every unit-bearing equation, call require_compatible(left, right, context) before the
   SymPy solve/dsolve call. Use require_dimension for individual inputs when useful.
5. If trigonometry is present, declare angle_mode and call angle_to_radians before sin/cos/tan.
6. Finish the script with exactly one call:
   emit_result(symbolic_solution=..., numerical_value=<number or Pint Quantity>,
               target_units=<unit string or None>, verification=[...])
   The worker, not you, applies significant figures and engineering notation.
7. If the tool reports an error, correct only the script or parsed assumptions and call the tool
   again. You have a bounded retry budget. Never invent a result to bypass an error.
8. After a successful tool result, explain the verified solution pedagogically. Quote the exact
   symbolic_solution, numerical_result, and units returned by the tool. Introduce no new numeric
   values beyond the original problem and verified output. State assumptions and unit checks.
9. Do not reveal hidden reasoning or claim the model independently verified the mathematics.
   Describe the observable parse, equations, tool checks, and verified output only.

SCRIPT SUBSET
- Allowed modules: sympy, pint, math.
- No files, network, subprocesses, dynamic imports, eval/exec, lambdify, classes, or functions.
- Use explicit SymPy constructors (symbols, Function, Eq, solve, dsolve, integrate, diff, Matrix).
- Pint quantities must be constructed with Q_(value, "unit") or value * ureg.unit.
"""


TOOL_NAME = "execute_verified_math"
TOOL_DEFINITION = {
    "name": TOOL_NAME,
    "description": (
        "Execute AST-validated SymPy/Pint code in a timeout-limited worker. This is the only "
        "permitted way to calculate or convert engineering mathematics."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "problem_analysis", "script", "requires_units", "angle_mode",
            "significant_figures",
        ],
        "properties": {
            "problem_analysis": {
                "type": "object",
                "additionalProperties": False,
                "required": ["knowns", "unknowns", "governing_equations", "assumptions"],
                "properties": {
                    "knowns": {"type": "array", "items": {"type": "string"}},
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "governing_equations": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                },
            },
            "script": {"type": "string", "minLength": 1, "maxLength": 32000},
            "requires_units": {"type": "boolean"},
            "angle_mode": {
                "type": "string", "enum": ["degrees", "radians", "unspecified"]
            },
            "significant_figures": {"type": "integer", "minimum": 2, "maximum": 8},
        },
    },
}


class ProblemAnalysis(BaseModel):
    knowns: list[str] = Field(min_length=1, max_length=30)
    unknowns: list[str] = Field(min_length=1, max_length=12)
    governing_equations: list[str] = Field(min_length=1, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("knowns", "unknowns", "governing_equations", "assumptions")
    @classmethod
    def nonempty_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Analysis entries must be non-empty strings.")
        return cleaned


class MathToolArguments(BaseModel):
    problem_analysis: ProblemAnalysis
    script: str = Field(min_length=1, max_length=32_000)
    requires_units: bool
    angle_mode: Literal["degrees", "radians", "unspecified"]
    significant_figures: int = Field(ge=2, le=8)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ModelTurn(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        require_tool: bool,
        allow_tools: bool,
    ) -> ModelTurn: ...


class SolveOutcome(BaseModel):
    request_id: str
    answer: str
    analysis: ProblemAnalysis
    verification: SandboxResult
    tool_attempts: int
    provider: str
    model: str
    trace: list[str]


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleProvider:
    """OpenAI Chat Completions adapter, including compatible base URLs."""

    provider_name = "openai-compatible"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=45.0)
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        require_tool: bool,
        allow_tools: bool,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": 0,
        }
        if allow_tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": TOOL_DEFINITION["name"],
                    "description": TOOL_DEFINITION["description"],
                    "parameters": TOOL_DEFINITION["input_schema"],
                    "strict": True,
                },
            }]
            kwargs["tool_choice"] = "required" if require_tool else "auto"
        with self._lock:
            response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"Provider returned invalid tool JSON: {exc.msg}.") from exc
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return ModelTurn(text=message.content or "", tool_calls=calls)


class OpenAIResponsesProvider:
    """Responses API adapter used by the packaged Muse desktop integration."""

    provider_name = "responses-compatible"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            if not api_key:
                raise PipelineError("Responses provider requires an API key.", status_code=503)
            client = OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=90.0,
                max_retries=0,
            )
        self.model = model
        self._client = client
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        require_tool: bool,
        allow_tools: bool,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": _to_responses_input(messages),
            "max_output_tokens": 4096,
            "reasoning": {"effort": "low"},
            "parallel_tool_calls": False,
        }
        if allow_tools:
            kwargs["tools"] = [{
                "type": "function",
                "name": TOOL_DEFINITION["name"],
                "description": TOOL_DEFINITION["description"],
                "parameters": TOOL_DEFINITION["input_schema"],
                "strict": True,
            }]
            # Muse's Responses API currently rejects `required` and named
            # function choices. Its only supported tool-selection value is
            # `auto`. The pipeline still enforces exactly one math call before
            # it will execute or release any answer, so this provider
            # compatibility setting does not weaken the verification gate.
            kwargs["tool_choice"] = "auto"
        with self._lock:
            response = self._client.responses.create(**kwargs)
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", "") != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"Provider returned invalid tool JSON: {exc.msg}.") from exc
            calls.append(ToolCall(
                id=item.call_id,
                name=item.name,
                arguments=arguments,
            ))
        return ModelTurn(text=response.output_text or "", tool_calls=calls)


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, *, api_key: str, model: str) -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key, timeout=45.0)
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        require_tool: bool,
        allow_tools: bool,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "messages": _to_anthropic_messages(messages),
            "temperature": 0,
            "max_tokens": 4096,
        }
        if allow_tools:
            kwargs["tools"] = [TOOL_DEFINITION]
            kwargs["tool_choice"] = {"type": "any" if require_tool else "auto"}
        with self._lock:
            response = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
        return ModelTurn(text="\n".join(text_parts), tool_calls=calls)


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            converted.append({
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], separators=(",", ":")),
                        },
                    }
                    for call in message["tool_calls"]
                ],
            })
        elif role == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": message["content"],
            })
        else:
            converted.append({"role": role, "content": message["content"]})
    return converted


def _to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            if message.get("content"):
                converted.append({"role": "assistant", "content": message["content"]})
            converted.extend(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], separators=(",", ":")),
                }
                for call in message["tool_calls"]
            )
        elif role == "tool":
            converted.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": message["content"],
            })
        else:
            converted.append({"role": role, "content": message["content"]})
    return converted


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                }
                for call in message["tool_calls"]
            )
            converted.append({"role": "assistant", "content": content})
        elif role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                    "is_error": message.get("is_error", False),
                }],
            })
        else:
            converted.append({"role": role, "content": message["content"]})
    return converted


def build_provider_from_env() -> ModelProvider:
    provider = os.getenv("MATH_PROVIDER", "openai").strip().lower()
    model = os.getenv("MATH_MODEL", "").strip()
    if provider in {"openai", "openai-compatible"}:
        api_key = os.getenv("OPENAI_API_KEY", os.getenv("MATH_API_KEY", "")).strip()
        if not api_key or not model:
            raise PipelineError(
                "Set MATH_MODEL and OPENAI_API_KEY (or MATH_API_KEY) before using /api/solve.",
                status_code=503,
            )
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL", os.getenv("MATH_BASE_URL", "")).strip() or None,
        )
    if provider in {"responses", "responses-compatible", "meta"}:
        api_key = os.getenv(
            "MATH_API_KEY",
            os.getenv("MODEL_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        ).strip()
        resolved_model = model or os.getenv("META_MODEL", "").strip()
        if not api_key or not resolved_model:
            raise PipelineError(
                "Set MATH_MODEL (or META_MODEL) and MATH_API_KEY (or MODEL_API_KEY).",
                status_code=503,
            )
        return OpenAIResponsesProvider(
            api_key=api_key,
            model=resolved_model,
            base_url=os.getenv(
                "MATH_BASE_URL", os.getenv("META_BASE_URL", "")
            ).strip() or None,
        )
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", os.getenv("MATH_API_KEY", "")).strip()
        if not api_key or not model:
            raise PipelineError(
                "Set MATH_MODEL and ANTHROPIC_API_KEY (or MATH_API_KEY) before using /api/solve.",
                status_code=503,
            )
        return AnthropicProvider(api_key=api_key, model=model)
    raise PipelineError(
        "MATH_PROVIDER must be openai, openai-compatible, responses-compatible, meta, or anthropic.",
        status_code=503,
    )


def _tool_payload(result: SandboxResult, analysis: ProblemAnalysis) -> str:
    return json.dumps(
        {"analysis": analysis.model_dump(), "sandbox": result.model_dump()},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _canonical_assistant(turn: ModelTurn) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": turn.text,
        "tool_calls": [call.model_dump() for call in turn.tool_calls],
    }


def _numbers(text: str) -> set[str]:
    return {
        _normalize_number(match)
        for match in re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    }


def _normalize_number(value: str) -> str:
    try:
        return format(float(value), ".12g")
    except ValueError:
        return value


def validate_grounded_answer(problem: str, answer: str, result: SandboxResult) -> list[str]:
    failures: list[str] = []
    if result.status != "success" or not result.numerical_result:
        return ["No successful verified result exists."]
    if result.numerical_result not in answer:
        failures.append("Answer does not quote the exact verified numerical_result.")
    if result.units and not re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(result.units)}(?![A-Za-z0-9_])", answer
    ):
        failures.append("Answer does not quote the exact verified units.")
    if "emit_result(" in answer or "import sympy" in answer or "import pint" in answer:
        failures.append("Answer leaked executable tool code.")
    allowed_numbers = _numbers(problem)
    allowed_numbers.update(_numbers(result.symbolic_solution or ""))
    allowed_numbers.update(_numbers(result.numerical_result or ""))
    unexpected = sorted(_numbers(answer) - allowed_numbers)
    if unexpected:
        failures.append("Answer introduced unverified numeric values: " + ", ".join(unexpected))
    return failures


def deterministic_fallback(analysis: ProblemAnalysis, result: SandboxResult) -> str:
    assumptions = "; ".join(analysis.assumptions) or "No additional assumptions were supplied."
    checks = " ".join(result.verification) or "The sandbox completed its required checks."
    unit_text = f" {result.units}" if result.units else ""
    return (
        "The deterministic solver completed successfully.\n\n"
        f"Knowns: {'; '.join(analysis.knowns)}\n\n"
        f"Unknowns: {'; '.join(analysis.unknowns)}\n\n"
        f"Governing equations: {'; '.join(analysis.governing_equations)}\n\n"
        f"Symbolic solution: {result.symbolic_solution}\n\n"
        f"Verified numerical result: {result.numerical_result}{unit_text}\n\n"
        f"Unit and execution checks: {checks}\n\n"
        f"Assumptions: {assumptions}"
    )


@dataclass
class MathAgentPipeline:
    provider: ModelProvider
    provider_name: str
    model: str
    max_tool_attempts: int = 3
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        require_tool: bool,
        allow_tools: bool,
    ) -> ModelTurn:
        try:
            return self.provider.complete(
                messages, require_tool=require_tool, allow_tools=allow_tools
            )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Model provider request failed ({type(exc).__name__}).",
                status_code=502,
            ) from exc

    def solve(
        self,
        *,
        problem: str,
        discipline: str,
        significant_figures: int,
    ) -> SolveOutcome:
        request_id = uuid.uuid4().hex
        trace = ["Problem accepted", "Provider asked to parse and call deterministic math tool"]
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"Discipline: {discipline}\n"
                f"Required significant figures: {significant_figures}\n"
                f"Engineering problem:\n{problem}"
            ),
        }]
        successful: SandboxResult | None = None
        analysis: ProblemAnalysis | None = None
        attempts = 0

        while attempts < self.max_tool_attempts:
            turn = self._complete(messages, require_tool=True, allow_tools=True)
            messages.append(_canonical_assistant(turn))
            if len(turn.tool_calls) != 1:
                raise PipelineError("Provider must issue exactly one deterministic math tool call.")
            call = turn.tool_calls[0]
            if call.name != TOOL_NAME:
                raise PipelineError(f"Unrecognized tool call: {call.name}.")
            attempts += 1
            try:
                arguments = MathToolArguments.model_validate(call.arguments)
            except ValidationError as exc:
                error_result = SandboxResult(
                    status="validation_error",
                    error_message="Invalid tool arguments: " + str(exc),
                    execution_id=request_id,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": error_result.model_dump_json(),
                    "is_error": True,
                })
                trace.append(f"Tool attempt {attempts} rejected: invalid arguments")
                continue

            analysis = arguments.problem_analysis
            # The HTTP request owns significant-figure policy; the model cannot override it.
            result = execute_math_script(
                arguments.script,
                requires_units=arguments.requires_units,
                angle_mode=arguments.angle_mode,
                significant_figures=significant_figures,
                limits=self.sandbox_limits,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": _tool_payload(result, analysis),
                "is_error": result.status != "success",
            })
            trace.append(f"Tool attempt {attempts}: {result.status}")
            if result.status == "success":
                successful = result
                break
            messages.append({
                "role": "user",
                "content": (
                    "Correct the failed script using the exact error above. Call the same tool "
                    "again; do not provide a prose answer or guess a result."
                ),
            })

        if successful is None or analysis is None:
            raise PipelineError(
                f"Deterministic solver did not succeed after {attempts} bounded attempts."
            )

        trace.append("Verified output returned to provider for explanation")
        messages.append({
            "role": "user",
            "content": (
                "Using only the successful tool output, write a concise step-by-step explanation. "
                "Include the parse, governing equation, symbolic result, unit verification, exact "
                "numerical result, assumptions, and one physical interpretation. Do no new math."
            ),
        })
        final_turn = self._complete(messages, require_tool=False, allow_tools=False)
        answer = final_turn.text.strip()
        failures = validate_grounded_answer(problem, answer, successful)
        if failures:
            trace.append("First explanation rejected by deterministic grounding gate")
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": (
                    "Rewrite the explanation without new calculations. Fix these release-gate "
                    "failures and quote the exact tool result: " + "; ".join(failures)
                ),
            })
            corrected = self._complete(messages, require_tool=False, allow_tools=False)
            answer = corrected.text.strip()
            failures = validate_grounded_answer(problem, answer, successful)
        if failures:
            trace.append("Provider explanation replaced with deterministic fallback")
            answer = deterministic_fallback(analysis, successful)
        else:
            trace.append("Grounded explanation released")

        return SolveOutcome(
            request_id=request_id,
            answer=answer,
            analysis=analysis,
            verification=successful,
            tool_attempts=attempts,
            provider=self.provider_name,
            model=self.model,
            trace=trace,
        )
