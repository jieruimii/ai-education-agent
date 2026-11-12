"""FastAPI boundary for the deterministic engineering math agent."""

from __future__ import annotations

import asyncio
import hmac
import os
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_pipeline import (
    MathAgentPipeline,
    PipelineError,
    SolveOutcome,
    build_provider_from_env,
)
from .examples import RLC_PROBLEM


RLC_EXAMPLE = {
    "discipline": "Electrical Engineering",
    "significant_figures": 4,
    "problem": RLC_PROBLEM,
}


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    problem: str = Field(min_length=20, max_length=12_000)
    discipline: Literal[
        "Mechanical Engineering",
        "Electrical Engineering",
        "Civil Engineering",
        "Chemical Engineering",
        "Aerospace Engineering",
        "General Engineering",
    ] = "General Engineering"
    significant_figures: int = Field(default=4, ge=2, le=8)

    @field_validator("problem")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise ValueError("Problem contains unsupported control characters.")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "deterministic-math-verification"
    provider_configured: bool


def _provider_is_configured() -> bool:
    provider = os.getenv("MATH_PROVIDER", "openai").strip().lower()
    model = bool(os.getenv("MATH_MODEL", "").strip())
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", os.getenv("MATH_API_KEY", "")).strip()
    else:
        key = os.getenv("OPENAI_API_KEY", os.getenv("MATH_API_KEY", "")).strip()
    return bool(model and key)


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = os.getenv("BACKEND_API_KEY", "").strip()
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


@lru_cache(maxsize=1)
def get_pipeline() -> MathAgentPipeline:
    provider = build_provider_from_env()
    return MathAgentPipeline(
        provider=provider,
        provider_name=getattr(provider, "provider_name", "configured-provider"),
        model=getattr(provider, "model", os.getenv("MATH_MODEL", "unknown")),
        max_tool_attempts=_bounded_env_int("MATH_MAX_TOOL_ATTEMPTS", 3, 1, 4),
    )


app = FastAPI(
    title="Deterministic Symbolic Math & Unit Verification Engine",
    version="1.0.0",
    description=(
        "Engineering problem parsing with model tool calls, followed by deterministic "
        "SymPy/Pint execution and a grounded explanation release gate."
    ),
)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_request: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(provider_configured=_provider_is_configured())


@app.get("/api/examples/rlc", dependencies=[Depends(require_api_key)])
async def rlc_example() -> dict[str, object]:
    """Return a ready-to-submit second-order RLC verification request."""

    return {
        "request": RLC_EXAMPLE,
        "submit_to": "/api/solve",
        "expected_method": (
            "SymPy dsolve/solve for the symbolic response, Pint dimensional checks, then "
            "numerical substitution and engineering-format output."
        ),
    }


@app.post(
    "/api/solve",
    response_model=SolveOutcome,
    dependencies=[Depends(require_api_key)],
)
async def solve(request: SolveRequest) -> SolveOutcome:
    pipeline = get_pipeline()
    return await asyncio.to_thread(
        pipeline.solve,
        problem=request.problem,
        discipline=request.discipline,
        significant_figures=request.significant_figures,
    )
