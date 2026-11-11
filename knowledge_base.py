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


SUPERVISED_VS_UNSUPERVISED = ConceptCard(
    card_id="ml.supervised-vs-unsupervised.v1",
    title="Supervised versus unsupervised machine learning",
    aliases=(
        "supervised machine learning and unsupervised machine learning",
        "unsupervised machine learning and supervised machine learning",
        "supervised vs unsupervised",
        "unsupervised vs supervised",
        "difference between unsupervised machine learning and supervised machine learning",
        "difference between supervised machine learning and unsupervised machine learning",
    ),
    claims=(
        "Supervised learning trains on input examples paired with target labels and learns a mapping that predicts the target for new inputs.",
        "Classification predicts a category, while regression predicts a numeric value; both are common supervised-learning tasks.",
        "Unsupervised learning receives inputs without target labels and searches for useful structure such as clusters, lower-dimensional representations, density patterns, or anomalies.",
        "The defining distinction is whether a target label is supplied for the learning objective, not whether a human ever touched the data.",
        "Supervised performance can be measured against held-out labels. Unsupervised results require task-specific validation because there may be no single ground-truth target.",
        "Neither approach is universally better; the choice follows the available data and the question being answered.",
    ),
    equations=(
        "Supervised empirical risk: minimize (1/n) Σ L(f(x_i), y_i), where y_i is the provided target label.",
        "A clustering example: choose cluster assignments and centers that minimize within-cluster distance; no target y_i is provided.",
    ),
    assumptions=(
        "training and evaluation data represent the intended deployment setting",
        "labels in supervised learning are sufficiently accurate and relevant",
        "the chosen unsupervised objective corresponds to structure that is useful for the real task",
    ),
    misconceptions=(
        "Unsupervised learning does not mean that no human choices or evaluation are involved.",
        "Unsupervised learning is not automatically more objective; its features, distance metric, and number of clusters encode choices.",
        "A dataset can support both approaches depending on whether a target variable is used.",
    ),
    example=(
        "Given emails labeled spam or not spam, learning to predict the label is supervised classification. "
        "Given unlabeled customer behavior and grouping similar customers, the task is unsupervised clustering."
    ),
    source_title="Stanford Engineering Everywhere, CS229 Machine Learning",
    source_url="https://see.stanford.edu/Course/CS229",
)


NEWTON_FIRST_LAW = ConceptCard(
    card_id="physics.newton-first-law.v1",
    title="Newton's first law of motion",
    aliases=("newton's 1st law", "newtons 1st law", "newton's first law", "newtons first law", "law of inertia"),
    claims=(
        "An object remains at rest or continues with constant velocity in a straight line unless a nonzero net external force acts on it.",
        "The law describes inertial reference frames and is often called the law of inertia.",
        "Constant velocity includes zero velocity, so rest and uniform straight-line motion are the same kind of state dynamically.",
        "Forces may be present and cancel; the law requires zero net force, not zero individual forces.",
        "A change in speed or direction is acceleration and therefore requires a nonzero net force in an inertial frame.",
    ),
    equations=("If ΣF_ext = 0, then a = 0 and velocity v is constant.",),
    assumptions=("an inertial reference frame", "the object is modeled with a well-defined net external force"),
    misconceptions=(
        "Motion does not require a continuing net force; a net force is required to change velocity.",
        "Zero net force does not mean no forces are present; balanced forces sum to zero.",
    ),
    example=("A book at rest has gravity downward and a normal force upward; the forces balance, so its acceleration is zero."),
    source_title="OpenStax University Physics, Newton's First Law",
    source_url="https://openstax.org/books/university-physics-volume-1/pages/5-2-newtons-first-law",
)


CONCEPT_CARDS = (THROTTLING, SUPERVISED_VS_UNSUPERVISED, NEWTON_FIRST_LAW)


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

