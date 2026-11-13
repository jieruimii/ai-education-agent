from __future__ import annotations

import unittest

from backend.math_verification.agent_pipeline import (
    MathAgentPipeline,
    ModelTurn,
    OpenAIResponsesProvider,
    ToolCall,
)
from backend.math_verification.math_sandbox import SandboxLimits


ANALYSIS = {
    "knowns": ["constant 1", "right-hand side 3"],
    "unknowns": ["x"],
    "governing_equations": ["x + 1 = 3"],
    "assumptions": ["x is real"],
}


class ScriptedProvider:
    provider_name = "test"
    model = "deterministic-fixture"

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = turns
        self.calls = 0

    def complete(self, messages, *, require_tool, allow_tools):
        self.calls += 1
        return self.turns.pop(0)


class FakeResponsesAPI:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeResponsesClient:
    def __init__(self, response) -> None:
        self.responses = FakeResponsesAPI(response)


class FakeFunctionCall:
    type = "function_call"
    call_id = "call-math"
    name = "execute_verified_math"
    arguments = '{"script":"value"}'


class FakeResponse:
    output = [FakeFunctionCall()]
    output_text = ""


def tool_turn(call_id: str, script: str) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall(
        id=call_id,
        name="execute_verified_math",
        arguments={
            "problem_analysis": ANALYSIS,
            "script": script,
            "requires_units": False,
            "angle_mode": "unspecified",
            "significant_figures": 4,
        },
    )])


class AgentPipelineTests(unittest.TestCase):
    def test_responses_adapter_forces_and_parses_math_tool(self) -> None:
        client = FakeResponsesClient(FakeResponse())
        provider = OpenAIResponsesProvider(client=client, model="muse-test")
        turn = provider.complete(
            [{"role": "user", "content": "Solve the equation"}],
            require_tool=True,
            allow_tools=True,
        )
        self.assertEqual(turn.tool_calls[0].id, "call-math")
        request = client.responses.calls[0]
        self.assertEqual(request["tool_choice"], "auto")
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["input"][0]["role"], "user")

    def test_failed_script_is_self_healed_then_grounded(self) -> None:
        invalid = '''
import os
import sympy as sp
x = sp.symbols("x")
solution = sp.solve(sp.Eq(x + 1, 3), x)[0]
emit_result(symbolic_solution=solution, numerical_value=solution)
'''
        valid = '''
import sympy as sp
x = sp.symbols("x")
solution = sp.solve(sp.Eq(x + 1, 3), x)[0]
emit_result(symbolic_solution=solution, numerical_value=solution)
'''
        final = (
            "The governing equation is x + 1 = 3. SymPy returned the symbolic solution 2, "
            "and the verified numerical result is 2.000. The result is dimensionless."
        )
        provider = ScriptedProvider([
            tool_turn("call-1", invalid),
            tool_turn("call-2", valid),
            ModelTurn(text=final),
        ])
        pipeline = MathAgentPipeline(
            provider=provider,
            provider_name=provider.provider_name,
            model=provider.model,
            sandbox_limits=SandboxLimits(timeout_seconds=12, cpu_seconds=10),
        )
        outcome = pipeline.solve(
            problem="Solve x + 1 = 3 for x using exact symbolic algebra.",
            discipline="General Engineering",
            significant_figures=4,
        )
        self.assertEqual(outcome.tool_attempts, 2)
        self.assertEqual(outcome.verification.status, "success")
        self.assertEqual(outcome.verification.numerical_result, "2.000")
        self.assertEqual(outcome.answer, final)
        self.assertIn("Tool attempt 1: validation_error", outcome.trace)


if __name__ == "__main__":
    unittest.main()
