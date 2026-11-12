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
        "supervised and unsupervised machine learning",
        "unsupervised and supervised machine learning",
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


INTEGRATION_BY_SUBSTITUTION = ConceptCard(
    card_id="math.integration-by-substitution.v1",
    title="Integration by substitution",
    aliases=(
        "integration by substitution",
        "u substitution",
        "u-substitution",
        "integral from 0 to 1 of x times exp x squared",
        "integral from 0 to 1 of x times exp(x squared)",
        "integral from 0 to 1 of x e x squared",
    ),
    claims=(
        "Substitution reverses the chain rule by replacing an inner expression with a new variable and replacing its differential at the same time.",
        "For a definite integral, changing variables also requires changing the limits to the new variable, unless the original variable is restored before endpoint evaluation.",
        "A proposed antiderivative can be checked by differentiating it and confirming that the original integrand is recovered.",
        "For the example integral from 0 to 1 of x times exp(x squared), the substitution u=x squared gives x dx=du/2 and the exact value (e-1)/2.",
    ),
    equations=(
        "Substitution rule: integral from a to b of f(g(x))g'(x) dx equals the integral from g(a) to g(b) of f(u) du.",
        "Example: integral from 0 to 1 of x exp(x squared) dx = (1/2) integral from 0 to 1 of exp(u) du = (e-1)/2.",
    ),
    assumptions=(
        "the substituted functions satisfy the continuity conditions of the substitution rule",
        "the transformed differential and definite-integral bounds are changed consistently",
    ),
    misconceptions=(
        "Do not divide by a variable derivative as though it were a constant; transform the full differential.",
        "Do not change the integrand to u while leaving definite-integral bounds expressed in x.",
    ),
    example=(
        "Let u=x squared in the integral from 0 to 1 of x exp(x squared) dx. Then du=2x dx, "
        "so the integral becomes one half times the integral from 0 to 1 of exp(u) du, giving (e-1)/2."
    ),
    source_title="OpenStax Calculus Volume 1, Section 5.5: Substitution",
    source_url="https://openstax.org/books/calculus-volume-1/pages/5-5-substitution",
)


SN1_VS_SN2 = ConceptCard(
    card_id="chem.sn1-vs-sn2.v1",
    title="SN1 versus SN2 nucleophilic substitution",
    aliases=(
        "sn1 and sn2",
        "sn2 and sn1",
        "sn1 vs sn2",
        "sn2 vs sn1",
        "compare sn1 and sn2",
        "compare sn1 and sn2 substitution reactions",
    ),
    claims=(
        "SN1 proceeds stepwise through rate-limiting ionization to a carbocation, followed by nucleophile attack; its rate depends primarily on substrate concentration.",
        "SN2 is a one-step concerted backside attack in which bond formation and leaving-group departure occur together; its rate depends on both substrate and nucleophile concentrations.",
        "SN1 is favored by substrates that form stable carbocations, especially tertiary substrates, and by polar protic solvents.",
        "SN2 is favored by unhindered methyl or primary substrates, strong nucleophiles, and polar aprotic solvents.",
        "SN2 gives inversion at a reacting stereocenter, whereas an idealized planar SN1 carbocation can be attacked from either face and commonly gives substantial racemization.",
    ),
    equations=(
        "SN1 rate law: rate = k[substrate].",
        "SN2 rate law: rate = k[substrate][nucleophile].",
    ),
    assumptions=(
        "the comparison concerns nucleophilic substitution rather than competing elimination",
        "substrate structure, nucleophile strength, leaving group, solvent, and temperature jointly affect the dominant pathway",
    ),
    misconceptions=(
        "Do not choose a mechanism from substrate class alone; solvent, nucleophile, leaving group, and elimination competition matter.",
        "SN1 is not guaranteed to produce a perfectly 50:50 racemic mixture because ion pairing and the leaving group can influence approach.",
    ),
    example=(
        "Tert-butyl halide solvolysis in water is a typical SN1 case; methyl bromide reacting with a strong nucleophile in a polar aprotic solvent is a typical SN2 case."
    ),
    source_title="Chemistry LibreTexts, SN1 vs SN2",
    source_url="https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_I_%28Liu%29/07%3A_Nucleophilic_Substitution_Reactions/7.05%3A_SN1_vs_SN2",
)


DRAMATIC_IRONY = ConceptCard(
    card_id="literature.dramatic-irony-oedipus.v1",
    title="Dramatic irony in Oedipus Rex",
    aliases=(
        "dramatic irony",
        "dramatic irony in oedipus rex",
        "oedipus rex dramatic irony",
        "dramatic irony in oedipus",
    ),
    claims=(
        "Dramatic irony occurs when the audience knows an important truth that a character does not know.",
        "In Oedipus Rex, the audience understands that Oedipus is implicated in the identity and crimes he investigates before he recognizes that truth himself.",
        "Oedipus's confident declarations and curse therefore carry a second meaning for the audience and intensify suspense, pity, and fear.",
        "A strong literary explanation identifies a specific knowledge gap and analyzes how it changes the meaning or effect of a line or action.",
    ),
    equations=("No governing equation applies; support the interpretation with the play's language and dramatic context.",),
    assumptions=(
        "the reader or audience knows the relevant mythic or dramatic information before the character recognizes it",
        "interpretive claims should be tied to specific moments in the text rather than treated as plot summary alone",
    ),
    misconceptions=(
        "Dramatic irony is not merely an unexpected event; it depends on a knowledge gap between audience and character.",
        "An analysis should explain the effect of the irony, not only identify that the audience knows more.",
    ),
    example=(
        "When Oedipus curses Laius's unknown killer, the audience understands that his words unknowingly condemn himself, giving the speech a meaning Oedipus cannot yet perceive."
    ),
    source_title="Gallaudet University Guide to Drama",
    source_url="https://gallaudet.edu/student-success/tutorial-center/english-center/writing/guides-to-understanding-and-writing-different-types-of-literature/guide-to-drama/",
)


FRENCH_REVOLUTION_CAUSES = ConceptCard(
    card_id="history.french-revolution-causes.v1",
    title="Causes of the French Revolution",
    aliases=(
        "causes of the french revolution",
        "main causes of the french revolution",
        "french revolution causes",
        "why did the french revolution happen",
    ),
    claims=(
        "The French Revolution resulted from interacting social, political, intellectual, and fiscal pressures rather than one isolated cause.",
        "The estate system concentrated privilege and wealth while much of the labor and tax burden fell on the Third Estate.",
        "Enlightenment and Atlantic-revolutionary ideas supplied language for criticizing inherited privilege and demanding rights and representation.",
        "State debt and the monarchy's inability to reform taxation produced a fiscal crisis that forced political action.",
        "The calling of the Estates General in 1789 created the immediate political opening in which accumulated grievances challenged royal authority.",
    ),
    equations=("No governing equation applies; distinguish long-term structural causes from short-term triggers and consequences.",),
    assumptions=(
        "historical causation is multi-causal and the relative weight assigned to each factor is interpretive",
        "the Storming of the Bastille was an early revolutionary event, not the sole underlying cause",
    ),
    misconceptions=(
        "Do not reduce the Revolution to taxes, Enlightenment ideas, food prices, or a single event alone.",
        "Do not confuse a long-term condition with the immediate institutional trigger in 1789.",
    ),
    example=(
        "The privileged estate structure and fiscal weakness were long-term conditions; convening the Estates General in May 1789 was an immediate trigger that enabled organized political confrontation."
    ),
    source_title="OpenStax World History Volume 2, Section 7.3",
    source_url="https://openstax.org/books/world-history-volume-2/pages/7-3-revolutions-america-france-and-haiti",
)


CONCEPT_CARDS = (
    THROTTLING,
    SUPERVISED_VS_UNSUPERVISED,
    NEWTON_FIRST_LAW,
    INTEGRATION_BY_SUBSTITUTION,
    SN1_VS_SN2,
    DRAMATIC_IRONY,
    FRENCH_REVOLUTION_CAUSES,
)


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

