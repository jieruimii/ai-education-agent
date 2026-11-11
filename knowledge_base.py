"""Small instructor-reviewable knowledge base for high-risk tutor concepts.

This is deliberately data, not a second agent.  The tutor harness retrieves a
matching card and gives its approved claims to the model as evidence.  Cards
are intentionally compact so an instructor can audit every claim before a
pilot.  They are not a substitute for a course-specific RAG collection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptCard:
    card_id: str
    title: str
    aliases: tuple[str, ...]
    claims: tuple[str, ...]
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    misconceptions: tuple[str, ...]
    example: str
    source_title: str
    source_url: str


THROTTLING = ConceptCard(
    card_id="thermo.throttling.v1",
    title="Thermodynamic throttling process",
    aliases=(
        "throttling process",
        "throttle process",
        "throttling valve",
        "expansion valve",
        "joule thomson",
        "pressure reducing valve",
    ),
    claims=(
        "A throttling device forces a flowing fluid through a restriction and produces a substantial pressure drop.",
        "For the standard steady, adiabatic, one-inlet/one-outlet model with no shaft work and negligible kinetic and potential energy changes, inlet and outlet specific enthalpy are equal.",
        "Throttling is irreversible. Entropy increases for the adiabatic idealization even though specific enthalpy remains approximately constant.",
        "Constant enthalpy does not generally mean constant temperature. The temperature change depends on the fluid and inlet state; an ideal gas with enthalpy depending only on temperature is a special case.",
        "Liquid refrigerant can partially flash into vapor after the pressure drop because the lower downstream pressure has a lower saturation temperature.",
        "A valve controls pressure or flow but does not extract useful shaft work as a turbine does.",
    ),
    equations=(
        "Steady-flow energy balance: q - w_s = (h_2-h_1) + (V_2^2-V_1^2)/2 + g(z_2-z_1).",
        "With q≈0, w_s=0, Δke≈0, and Δpe≈0: h_1≈h_2.",
        "For an adiabatic real throttle: s_2>s_1.",
    ),
    assumptions=(
        "steady state and one inlet/one outlet",
        "negligible heat transfer during the short residence time",
        "no shaft work",
        "negligible inlet-to-outlet kinetic and potential energy changes",
    ),
    misconceptions=(
        "Do not call the process isothermal merely because it is isenthalpic.",
        "Do not call it isentropic; a real pressure drop through a restriction is irreversible.",
        "Do not treat the pressure loss as useful work output.",
    ),
    example=(
        "In a vapor-compression refrigerator, high-pressure liquid refrigerant crosses an expansion valve. "
        "Its pressure and saturation temperature fall; at nearly unchanged enthalpy, part of the liquid flashes "
        "to vapor, producing a cold two-phase mixture for the evaporator."
    ),
    source_title="MIT Unified Engineering Thermodynamics, Chapter 4",
    source_url="https://ocw.mit.edu/ans7870/16/16.unified/thermoF03/chapter_4.htm",
)


CONCEPT_CARDS = (THROTTLING,)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def find_concept_cards(question: str, course: str = "") -> tuple[ConceptCard, ...]:
    """Return exact alias matches; avoid fuzzy retrieval that can inject wrong facts."""

    haystack = _normalized(f"{course} {question}")
    matches = []
    for card in CONCEPT_CARDS:
        if any(_normalized(alias) in haystack for alias in card.aliases):
            matches.append(card)
    return tuple(matches)


def approved_source_urls() -> set[str]:
    return {card.source_url for card in CONCEPT_CARDS}

