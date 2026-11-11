"""Deterministic educational logic for the Tkinter prototype.

The module deliberately has no UI or model-provider dependency.  It can be used
by local mock mode today and by a provider adapter later without changing the
pedagogical state machine.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class WorkflowStage(str, Enum):
    """Student-visible stages for hint-first problem solving."""

    DIAGNOSE = "Diagnose"
    DESIGN_HINT = "Design Hint"
    GATHER_REPORT = "Gather Report"
    ITERATE = "Iterate"


@dataclass
class StudentProfile:
    """Small, explicitly provided learner profile for the local prototype."""

    student_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    engineering_major: str = "General Engineering"
    academic_level: str = "Undergraduate"
    learning_goal: str = "Build conceptual understanding"
    # Retained only for compatibility with older saved/test objects. The UI no
    # longer exposes this field and routing must never use it.
    active_course: str = ""
    explanation_preference: str = "Step-by-step"


@dataclass(frozen=True)
class StudentAttempt:
    text: str
    quality: str


@dataclass
class ProblemSolvingSession:
    """Serializable state for one student's one problem-solving session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: WorkflowStage = WorkflowStage.DIAGNOSE
    problem: str = ""
    known_information: list[str] = field(default_factory=list)
    unknown_variables: list[str] = field(default_factory=list)
    attempted_approach: str = ""
    attempts: list[StudentAttempt] = field(default_factory=list)
    hints_given: list[str] = field(default_factory=list)
    hint_level: int = 0
    stuck_count: int = 0


@dataclass(frozen=True)
class EducationTurn:
    text: str
    stage: WorkflowStage | None = None
    flagged: bool = False
    reason: str = ""
    mock: bool = True


class EducationService(Protocol):
    """Provider-independent contract used by the Tkinter application."""

    def tutor_reply(self, profile: StudentProfile, message: str) -> EducationTurn: ...

    def problem_reply(
        self,
        profile: StudentProfile,
        session: ProblemSolvingSession,
        message: str,
    ) -> EducationTurn: ...


_DIRECT_REQUEST_PATTERNS = (
    r"\bgive\s+me\s+(?:the\s+)?(?:final\s+)?answer\b",
    r"\bjust\s+(?:solve|do)\s+(?:it|this)(?:\s+for\s+me)?\b",
    r"\bdo\s+it\s+for\s+me\b",
    r"\bwhat(?:'s|\s+is)\s+the\s+final\s+answer\b",
    r"\bignore\s+(?:the\s+policy|previous|all\s+instructions)\b",
    r"\bshow\s+me\s+the\s+complete\s+solution\b",
)

_OUTPUT_LEAKAGE_PATTERNS = (
    r"\bfinal\s+answer\s*(?:is|:|=)",
    r"\btherefore,?\s+(?:the\s+)?answer\s+(?:is|equals)\b",
    r"\bboxed\s*\{",
    r"\bcomplete\s+worked\s+solution\b",
)

_STUCK_PHRASES = (
    "i don't know",
    "i dont know",
    "no idea",
    "i'm stuck",
    "im stuck",
    "not sure",
    "confused",
    "help me",
    "idk",
)

_PROGRESS_CUES = (
    "because",
    "therefore",
    "equation",
    "balance",
    "sum",
    "moment",
    "force",
    "voltage",
    "current",
    "energy",
    "derivative",
    "integral",
    "substitute",
    "diagram",
    "i tried",
    "my approach",
)


def is_direct_answer_request(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return any(re.search(pattern, normalized) for pattern in _DIRECT_REQUEST_PATTERNS)


def contains_output_leakage(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in _OUTPUT_LEAKAGE_PATTERNS)


def safeguard_problem_solver_output(text: str) -> tuple[str, bool]:
    """Replace obvious leaked solutions with a safe next-step request.

    This is a narrow prototype safeguard, not a correctness or safety guarantee.
    """

    if not contains_output_leakage(text):
        return text, False
    return (
        "I withheld a response that appeared to reveal a complete answer. "
        "Show your latest equation, diagram, or reasoning step and I will give "
        "the smallest useful hint.",
        True,
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _diagnostic_evidence(message: str) -> tuple[bool, bool, bool]:
    text = message.lower()
    has_knowns = _has_any(text, ("known", "given", "have", "load", "value", "data"))
    has_unknown = _has_any(text, ("unknown", "find", "solve for", "determine", "need"))
    has_attempt = _has_any(text, ("tried", "attempt", "approach", "equation", "diagram", "started"))
    return has_knowns, has_unknown, has_attempt


def _is_stuck(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return _has_any(normalized, _STUCK_PHRASES)


def _is_substantive_attempt(message: str) -> bool:
    text = " ".join(message.lower().split())
    if len(text.split()) < 5:
        return False
    return bool(re.search(r"[=+\-*/]", text)) or _has_any(text, _PROGRESS_CUES)


def _attempt_quality(message: str) -> str:
    text = " ".join(message.lower().split())
    if _is_stuck(text):
        return "confused"
    if _has_any(text, ("got it", "makes sense", "checks out", "units match")):
        return "progress"
    if _is_substantive_attempt(text):
        return "partial"
    return "insufficient"


def _topic_key(text: str) -> str:
    normalized = text.lower()
    if _has_any(normalized, ("newton's first law", "newtons first law", "newton's 1st law", "newtons 1st law", "law of inertia")):
        return "newton first law"
    if _has_any(normalized, ("machine learning", "supervised", "unsupervised", "classification", "clustering")):
        return "machine learning"
    if _has_any(normalized, ("moment", "reaction", "beam", "free-body", "free body", "statics")):
        return "statics"
    if _has_any(normalized, ("circuit", "current", "voltage", "resistor", "kcl", "kvl")):
        return "circuits"
    if _has_any(normalized, (
        "thermo", "heat", "enthalpy", "energy balance", "control volume",
        "throttl", "expansion valve", "pressure reducing valve",
    )):
        return "thermodynamics"
    if _has_any(normalized, ("derivative", "integral", "calculus", "differentiate")):
        return "calculus"
    if _has_any(normalized, ("probability", "random variable", "bayes")):
        return "probability"
    return "engineering problem"


def select_hint_level(session: ProblemSolvingSession) -> int:
    """Select 1-3 from observed struggle and submitted attempts."""

    evidence = len(session.attempts)
    return min(3, 1 + session.stuck_count + evidence // 2)


def progressive_hint(topic: str, level: int) -> str:
    hints = {
        "statics": (
            "Draw an isolated free-body diagram and ask which forces and moments must balance.",
            "Use rigid-body equilibrium: choose a moment center that eliminates the most unknown reactions, then apply force balance.",
            "Write the symbolic setup ΣM_A = 0 first, followed by ΣF_x = 0 and ΣF_y = 0. Keep the reactions symbolic and stop before the final substitution.",
        ),
        "circuits": (
            "Identify whether the shared quantity is current or voltage before choosing a law.",
            "Apply Ohm's law to each element, then use KCL at a node or KVL around one loop for the next operation.",
            "Label branch currents and write one symbolic node or loop equation, such as ΣI_in = ΣI_out or ΣV = 0, without solving the final value.",
        ),
        "thermodynamics": (
            "Choose the system boundary and identify which forms of energy can cross it.",
            "Start from the first-law energy balance and cross out terms only after stating the steady-state and heat/work assumptions.",
            "Write the symbolic control-volume balance with inlet, outlet, heat, and work terms; substitute known terms but leave the requested value unsolved.",
        ),
        "calculus": (
            "Decide whether the question asks for a rate of change, accumulation, or an extremum.",
            "Name the differentiation or integration rule that matches the expression, then perform only the first transformation.",
            "Write the applicable rule symbolically and substitute the functions or limits, stopping before simplifying to the final result.",
        ),
        "probability": (
            "Define the event or random variable before selecting a formula.",
            "Write the sample space and decide whether conditioning, independence, or a named distribution applies.",
            "Set up the symbolic probability expression with the correct events and denominator, but do not evaluate the final number.",
        ),
        "engineering problem": (
            "Identify the system boundary, governing principle, knowns, and unknowns.",
            "Choose the conservation or constitutive relationship that connects the known quantities to the unknown.",
            "Write the governing equation symbolically and substitute known quantities, but stop before calculating the final result.",
        ),
    }
    return hints.get(topic, hints["engineering problem"])[max(1, min(level, 3)) - 1]


def _profile_framing(profile: StudentProfile) -> str:
    level = profile.academic_level.lower()
    if "first" in level or "high school" in level:
        depth = "I will start with physical intuition and define each symbol before using it."
    elif "graduate" in level and "undergraduate" not in level:
        depth = "I will emphasize assumptions, model limits, and generalization."
    else:
        depth = "I will connect physical intuition to the governing equation."

    major = profile.engineering_major.lower()
    if "electrical" in major:
        connection = "I will connect the idea to circuits, signals, or energy flow where useful."
    elif "mechanical" in major or "aerospace" in major:
        connection = "I will connect the idea to forces, motion, structures, or energy systems where useful."
    elif "civil" in major:
        connection = "I will connect the idea to loads, structures, and safety margins where useful."
    elif "computer" in major or "data" in major:
        connection = "I will connect the idea to algorithms, data, or computational models where useful."
    else:
        connection = "I will use a broadly applicable engineering example."
    return f"{depth} {connection}"


def _worked_example(topic: str) -> str:
    if topic == "newton first law":
        return (
            "Worked example: a book resting on a table has gravity downward and an equal normal force "
            "upward. The net external force is zero, so its acceleration is zero and it remains at rest."
        )
    if topic == "statics":
        return (
            "Worked practice example: a simply supported 4 m beam carries a centered 10 kN load. "
            "By symmetry, R_A = R_B. Vertical equilibrium gives R_A + R_B - 10 = 0, "
            "so 2R_A = 10 and R_A = R_B = 5 kN upward. A force check gives 5 + 5 - 10 = 0."
        )
    if topic == "circuits":
        return (
            "Worked practice example: two series resistors, 2 Ω and 4 Ω, are connected to 12 V. "
            "The equivalent resistance is 2 + 4 = 6 Ω. Ohm's law gives I = 12/6 = 2 A. "
            "The drops are 4 V and 8 V, and KVL checks: 4 + 8 = 12 V."
        )
    if topic == "thermodynamics":
        return (
            "Worked practice example: an insulated steady-flow device has h_in = 300 kJ/kg and "
            "h_out = 250 kJ/kg, with negligible kinetic and potential changes. The steady-flow "
            "energy balance gives w_out = h_in - h_out = 50 kJ/kg."
        )
    if topic == "calculus":
        return (
            "Worked practice example: for position x(t) = 3t² + 2t, velocity is the derivative, "
            "v(t) = 6t + 2. At t = 2 s, v = 14 m/s. Differentiating again gives constant "
            "acceleration a(t) = 6 m/s²."
        )
    if topic == "probability":
        return (
            "Worked practice example: if a component passes with probability 0.9 independently, "
            "two components both pass with probability 0.9 × 0.9 = 0.81. The complement shows "
            "the chance of at least one failure is 1 - 0.81 = 0.19."
        )
    if topic == "machine learning":
        return (
            "Worked comparison: predicting whether an email is spam from examples labeled spam/not-spam "
            "is supervised classification. Grouping unlabeled customers by similar behavior is unsupervised "
            "clustering. The key question is whether the training objective receives a target label y."
        )
    return (
        "Worked practice method: define the system, list knowns and unknowns, select the governing "
        "principle, solve symbolically, substitute units, and verify the result against a limiting case."
    )


def _throttling_explanation() -> str:
    """Instructor-reviewable mock answer for the meeting's regression case."""

    return (
        "A throttling process forces a flowing fluid through a restriction so its pressure drops; "
        "under the standard valve assumptions, its specific enthalpy stays approximately constant.\n\n"
        "## What it is\n"
        "A throttle can be an expansion valve, capillary tube, porous plug, or pressure-reducing valve. "
        "It controls pressure or flow. Unlike a turbine, it does not turn the pressure drop into useful shaft work.\n\n"
        "## How it works\n"
        "Upstream pressure pushes fluid through a narrow flow area. Viscous dissipation and mixing make "
        "the process irreversible. Pressure falls across the restriction, while the entering and leaving "
        "flow energy is accounted for through enthalpy. In refrigeration, the lower downstream saturation "
        "temperature can make some liquid refrigerant flash into vapor.\n\n"
        "## Governing equation\n"
        "For one inlet and one outlet, the steady-flow energy equation is "
        "$q-w_s=(h_2-h_1)+(V_2^2-V_1^2)/2+g(z_2-z_1)$. "
        "If heat transfer, shaft work, and kinetic/potential-energy changes are negligible, it reduces to "
        "$h_1\\approx h_2$. This is an isenthalpic—not necessarily isothermal—process.\n\n"
        "## Assumptions and limits\n"
        "The familiar $h_1\\approx h_2$ result assumes steady flow, one inlet and outlet, negligible heat "
        "transfer, no shaft work, and negligible kinetic and potential energy changes. If a nozzle-like "
        "velocity change or significant heat transfer exists, use the full energy balance.\n\n"
        "## Concrete example\n"
        "In a refrigerator, high-pressure liquid refrigerant crosses the expansion valve. Its pressure and "
        "saturation temperature fall. At nearly constant enthalpy, part of the liquid flashes, leaving a cold "
        "two-phase mixture that can absorb heat in the evaporator.\n\n"
        "## Common misconception\n"
        "Constant enthalpy does not guarantee constant temperature, and throttling is not isentropic. A real "
        "adiabatic throttle is irreversible, so entropy increases. An ideal gas is a special case: because its "
        "enthalpy depends mainly on temperature, the ideal-gas temperature is unchanged in this model.\n\n"
        "## Check your understanding\n"
        "Why can the refrigerant temperature fall after the valve even though $h_1\\approx h_2$?\n\n"
        "Source: MIT Unified Engineering Thermodynamics, Chapter 4 — "
        "https://ocw.mit.edu/ans7870/16/16.unified/thermoF03/chapter_4.htm"
    )


def _machine_learning_comparison() -> str:
    return (
        "Supervised learning learns to predict a supplied target label, while unsupervised learning looks "
        "for useful structure in inputs that have no target label.\n\n"
        "## What it is\n"
        "In supervised learning, every training example contains an input $x$ and a desired target $y$. "
        "In unsupervised learning, the algorithm receives $x$ values but no target $y$ for the objective.\n\n"
        "## How it works\n"
        "A supervised model adjusts its parameters so predictions match known labels, then is tested on "
        "held-out labeled examples. An unsupervised model instead optimizes a structural objective—such as "
        "making points within a cluster similar or compressing the data while preserving important variation.\n\n"
        "## Governing equation\n"
        "A common supervised objective is $\\min_f \\frac{1}{n}\\sum_i L(f(x_i),y_i)$, where $L$ measures "
        "prediction error. A clustering objective can minimize distances from points to assigned cluster "
        "centers; it contains no supplied target $y_i$.\n\n"
        "## Assumptions and limits\n"
        "Supervised learning depends on representative, accurate labels. Unsupervised learning depends on "
        "whether the chosen features, similarity measure, and objective capture structure useful to the real task.\n\n"
        "## Concrete example\n"
        "Learning from emails labeled spam/not-spam is supervised classification. Grouping unlabeled "
        "customers by behavioral similarity is unsupervised clustering.\n\n"
        "## Common misconception\n"
        "Unsupervised does not mean no humans are involved or that the result is automatically objective. "
        "Humans still choose the data, features, algorithm, distance metric, and evaluation method.\n\n"
        "## Check your understanding\n"
        "If you have customer records plus a churn/no-churn target, which approach would you use to predict churn—and why?\n\n"
        "Source: Stanford Engineering Everywhere, CS229 Machine Learning — "
        "https://see.stanford.edu/Course/CS229"
    )


def _newton_first_law_explanation() -> str:
    return (
        "Newton's first law says that an object's velocity stays constant—including remaining at rest—unless a nonzero net external force changes it.\n\n"
        "## What it is\nIt is the law of inertia: objects do not spontaneously change their speed or direction.\n\n"
        "## How it works\nIf all external forces balance, there is no acceleration. The object remains at rest or moves in a straight line at constant speed.\n\n"
        "## Governing equation\nIf $\\sum \\vec{F}_{ext}=0$, then $\\vec{a}=0$ and velocity $\\vec{v}$ is constant.\n\n"
        "## Assumptions and limits\nThis statement is made in an inertial reference frame. In an accelerating frame, apparent or fictitious forces may be needed.\n\n"
        "## Concrete example\nA book on a table has gravity downward and a normal force upward. They cancel, so the book has zero acceleration.\n\n"
        "## Common misconception\nZero net force does not mean no forces exist, and an object does not need a continuing force to keep moving at constant velocity.\n\n"
        "## Check your understanding\nA puck slides at constant velocity on nearly frictionless ice. What is the net horizontal force on it?\n\n"
        "Source: OpenStax University Physics, Newton's First Law — https://openstax.org/books/university-physics-volume-1/pages/5-2-newtons-first-law"
    )


def tutor_response(profile: StudentProfile, message: str) -> str:
    # The current question is authoritative. Profile fields personalize style,
    # but must never override or contaminate topic routing.
    topic = _topic_key(message)
    normalized = message.lower()
    if topic == "newton first law":
        return _newton_first_law_explanation()
    if topic == "thermodynamics" and any(
        term in normalized for term in ("throttl", "expansion valve", "pressure reducing valve")
    ):
        return _throttling_explanation()
    if topic == "machine learning" and "supervised" in normalized and "unsupervised" in normalized:
        return _machine_learning_comparison()
    preference = profile.explanation_preference.lower()
    style = (
        "I will keep the explanation compact."
        if "concise" in preference
        else "I will show the reasoning in labeled steps."
        if "step" in preference
        else "I will begin with an analogy, then connect it to the equation."
    )
    return (
        "Mock Tutor — deterministic local response.\n\n"
        f"Profile: {profile.academic_level} {profile.engineering_major}; goal: {profile.learning_goal}. "
        f"{_profile_framing(profile)} {style}\n\n"
        f"{_worked_example(topic)}\n\n"
        "Comprehension check: which assumption or equation would change first if one input or "
        "boundary condition changed?"
    )


def problem_solver_response(
    profile: StudentProfile,
    session: ProblemSolvingSession,
    message: str,
) -> EducationTurn:
    """Advance one deterministic turn without ever intentionally solving the problem."""

    direct_request = is_direct_answer_request(message)
    if direct_request:
        safe = (
            "I can’t provide the final answer in Problem-Solving mode. "
            "I can help you build it. List the known quantities, the unknown, and your first attempted step."
        )
        return EducationTurn(
            text=safe,
            stage=session.stage,
            flagged=True,
            reason="Potential answer leakage or policy bypass request",
        )

    if session.stage == WorkflowStage.DIAGNOSE:
        if not session.problem:
            session.problem = message.strip()
        has_knowns, has_unknown, has_attempt = _diagnostic_evidence(message)
        if has_knowns:
            session.known_information.append(message.strip())
        if has_unknown:
            session.unknown_variables.append(message.strip())
        if has_attempt:
            session.attempted_approach = message.strip()
        evidence_count = sum((has_knowns, has_unknown, has_attempt))
        if evidence_count < 2:
            response = (
                "Stage 1 — Diagnose. I have the problem, but I need a clearer diagnosis before giving a hint. "
                "Please state: (1) the known information, (2) the unknown variable, and "
                "(3) what you have tried or which governing principle might apply."
            )
        else:
            session.stage = WorkflowStage.DESIGN_HINT
            response = (
                "Stage 1 — Diagnose complete. I recorded your knowns, target unknown, and current approach. "
                "We are ready to design the smallest useful hint; ask for a hint or identify the exact step blocking you."
            )

    elif session.stage == WorkflowStage.DESIGN_HINT:
        if not (_is_stuck(message) or "hint" in message.lower() or _is_substantive_attempt(message)):
            response = (
                "Stage 2 — Design Hint. Tell me which step is blocking you or explicitly request a hint. "
                "I will not advance merely because another message was sent."
            )
        else:
            if _is_stuck(message):
                session.stuck_count += 1
            if _is_substantive_attempt(message):
                session.attempts.append(StudentAttempt(message.strip(), _attempt_quality(message)))
            session.hint_level = select_hint_level(session)
            topic = _topic_key(f"{session.problem} {message}")
            hint = progressive_hint(topic, session.hint_level)
            session.hints_given.append(hint)
            session.stage = WorkflowStage.GATHER_REPORT
            response = (
                f"Stage 2 — Design Hint (level {session.hint_level} of 3). {hint}\n\n"
                "Stage 3 — Gather Report: show the equation, diagram, or reasoning step you try next."
            )

    elif session.stage == WorkflowStage.GATHER_REPORT:
        if _is_stuck(message):
            session.stuck_count += 1
            session.stage = WorkflowStage.DESIGN_HINT
            response = (
                "Stage 3 — Gather Report. I recorded that you are still blocked. "
                "We will return to Design Hint and make the next hint more concrete."
            )
        elif not _is_substantive_attempt(message):
            response = (
                "Stage 3 — Gather Report. That does not yet show enough reasoning to evaluate. "
                "Please submit at least one equation, diagram description, governing principle, or explanatory step."
            )
        else:
            quality = _attempt_quality(message)
            session.attempts.append(StudentAttempt(message.strip(), quality))
            session.stage = WorkflowStage.ITERATE
            response = (
                "Stage 3 — Gather Report complete. Your attempt contains reasoning evidence. "
                "We can now inspect the choice of principle, signs, units, and assumptions."
            )

    else:  # ITERATE
        quality = _attempt_quality(message)
        if quality == "confused":
            session.stuck_count += 1
            session.stage = WorkflowStage.DESIGN_HINT
            response = (
                "Stage 4 — Iterate. The report shows continued confusion, so we are returning to "
                "Design Hint for a more concrete scaffold."
            )
        elif quality == "insufficient":
            session.stage = WorkflowStage.GATHER_REPORT
            response = (
                "Stage 4 — Iterate. I cannot evaluate progress from that response yet. "
                "Return to Gather Report and show the revised equation or reasoning step."
            )
        else:
            session.attempts.append(StudentAttempt(message.strip(), quality))
            session.stage = WorkflowStage.GATHER_REPORT
            response = (
                "Stage 4 — Iterate. Your approach shows progress. Before calculating further, check the "
                "sign convention, units, and whether every term belongs to the chosen system boundary. "
                "Then report the revised symbolic step; I will continue without revealing the final answer."
            )

    safe_response, blocked = safeguard_problem_solver_output(response)
    return EducationTurn(
        text=safe_response,
        stage=session.stage,
        flagged=blocked,
        reason="Generated response matched a final-answer leakage pattern" if blocked else "",
    )


class MockEducationService:
    """Deterministic local implementation used by demos and automated tests."""

    def tutor_reply(self, profile: StudentProfile, message: str) -> EducationTurn:
        return EducationTurn(text=tutor_response(profile, message))

    def problem_reply(
        self,
        profile: StudentProfile,
        session: ProblemSolvingSession,
        message: str,
    ) -> EducationTurn:
        return problem_solver_response(profile, session, message)

