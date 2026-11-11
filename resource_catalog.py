"""Curated, locally ranked public university resources.

The catalog is intentionally static: the prototype performs no crawling, remote
search, or content ingestion.  URLs and titles were verified on official
institution domains on 2026-11-05.  A verified URL does not imply permission to
redistribute its content; each record keeps a short license/terms reminder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from learning_core import StudentProfile


VERIFICATION_DATE = "2026-11-05"


@dataclass(frozen=True)
class LearningResource:
    institution: str
    title: str
    topics: tuple[str, ...]
    academic_levels: tuple[str, ...]
    engineering_majors: tuple[str, ...]
    url: str
    license_note: str
    verification_date: str
    verified: bool = True


@dataclass(frozen=True)
class ResourceRecommendation:
    resource: LearningResource
    score: int
    reasons: tuple[str, ...]


MIT_LICENSE = (
    "Open access through MIT OpenCourseWare; most OCW materials use CC BY-NC-SA, "
    "but verify the individual course terms before reuse."
)
STANFORD_LICENSE = (
    "Stanford Engineering Everywhere materials are generally CC BY-NC-SA 4.0; "
    "verify the individual course notice before reuse."
)
NPTEL_LICENSE = (
    "Free public course access through NPTEL; verify the individual resource's "
    "reuse and redistribution terms."
)
DELFT_LICENSE = "TU Delft OpenCourseWare course marked CC BY-NC-SA 4.0."


RESOURCES: tuple[LearningResource, ...] = (
    LearningResource(
        "MIT OpenCourseWare",
        "2.003SC Engineering Dynamics",
        ("engineering dynamics", "mechanics", "motion", "forces", "vibration"),
        ("Undergraduate",),
        ("Mechanical Engineering", "Aerospace Engineering", "Civil Engineering"),
        "https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/",
        MIT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "MIT OpenCourseWare",
        "6.002 Circuits and Electronics",
        ("circuits", "electronics", "voltage", "current", "resistors", "mosfet"),
        ("Undergraduate",),
        ("Electrical Engineering", "Computer Engineering"),
        "https://ocw.mit.edu/courses/6-002-circuits-and-electronics-spring-2007/",
        MIT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "MIT OpenCourseWare",
        "18.01SC Single Variable Calculus",
        ("calculus", "derivatives", "integrals", "rates", "optimization"),
        ("First-year Undergraduate", "Undergraduate"),
        ("General Engineering", "Mechanical Engineering", "Electrical Engineering", "Civil Engineering"),
        "https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/",
        MIT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "Stanford Engineering Everywhere",
        "CS229 Machine Learning",
        ("machine learning", "probability", "linear algebra", "optimization", "data"),
        ("Advanced Undergraduate", "Graduate"),
        ("Computer Engineering", "Data Science", "Electrical Engineering"),
        "https://see.stanford.edu/Course/CS229",
        STANFORD_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "Stanford Engineering Everywhere",
        "EE263 Introduction to Linear Dynamical Systems",
        ("linear systems", "dynamics", "control", "circuits", "signals", "linear algebra"),
        ("Advanced Undergraduate", "Graduate"),
        ("Electrical Engineering", "Mechanical Engineering", "Aerospace Engineering"),
        "https://see.stanford.edu/Course/EE263",
        STANFORD_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "Stanford Engineering Everywhere",
        "EE364A Convex Optimization I",
        ("convex optimization", "optimization", "control", "linear algebra", "engineering design"),
        ("Graduate",),
        ("Electrical Engineering", "Computer Engineering", "Data Science"),
        "https://see.stanford.edu/Course/EE364A",
        STANFORD_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "NPTEL — IIT Kanpur",
        "Engineering Mechanics",
        ("engineering mechanics", "statics", "dynamics", "forces", "equilibrium"),
        ("First-year Undergraduate", "Undergraduate"),
        ("Mechanical Engineering", "Civil Engineering", "Aerospace Engineering", "General Engineering"),
        "https://nptel.ac.in/courses/122104014",
        NPTEL_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "NPTEL — IIT Kharagpur",
        "Network Analysis",
        ("network analysis", "circuits", "kcl", "kvl", "laplace transform", "two-port networks"),
        ("Undergraduate",),
        ("Electrical Engineering", "Electronics Engineering"),
        "https://nptel.ac.in/courses/108105159",
        NPTEL_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "NPTEL — IIT Kharagpur",
        "Introduction to Fluid Mechanics",
        ("fluid mechanics", "flow", "pressure", "control volume", "transport phenomena"),
        ("Undergraduate",),
        ("Mechanical Engineering", "Civil Engineering", "Aerospace Engineering", "Chemical Engineering"),
        "https://nptel.ac.in/courses/112105269",
        NPTEL_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "NPTEL — IIT Kanpur",
        "Chemical Engineering Thermodynamics",
        ("thermodynamics", "energy", "entropy", "enthalpy", "phase equilibrium"),
        ("Undergraduate",),
        ("Chemical Engineering", "Mechanical Engineering"),
        "https://nptel.ac.in/courses/103104151",
        NPTEL_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "TU Delft OpenCourseWare",
        "Aerospace Mechanics of Materials",
        ("mechanics of materials", "statics", "stress", "strain", "beams", "structures"),
        ("Undergraduate",),
        ("Aerospace Engineering", "Mechanical Engineering", "Civil Engineering"),
        "https://ocw.tudelft.nl/courses/aerospace-mechanics-of-materials/",
        DELFT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "TU Delft OpenCourseWare",
        "Electrical Machines and Drives",
        ("electrical machines", "motors", "generators", "power electronics", "magnetic circuits"),
        ("Graduate",),
        ("Electrical Engineering", "Mechanical Engineering"),
        "https://ocw.tudelft.nl/courses/electrical-machines-and-drives/",
        DELFT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "TU Delft OpenCourseWare",
        "The Basics of Transport Phenomena",
        ("transport phenomena", "fluid mechanics", "heat transfer", "mass transfer", "thermodynamics"),
        ("Undergraduate",),
        ("Chemical Engineering", "Mechanical Engineering"),
        "https://ocw.tudelft.nl/courses/basics-transport-phenomena/",
        DELFT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "TU Delft OpenCourseWare",
        "Introduction to Aerospace Engineering I",
        ("aerospace", "aerodynamics", "flight mechanics", "structures", "control volume"),
        ("Undergraduate",),
        ("Aerospace Engineering", "Mechanical Engineering"),
        "https://ocw.tudelft.nl/courses/introduction-aerospace-engineering/",
        DELFT_LICENSE,
        VERIFICATION_DATE,
    ),
    LearningResource(
        "TU Delft OpenCourseWare",
        "Solar Energy",
        ("solar energy", "photovoltaics", "semiconductors", "energy conversion", "electrical systems"),
        ("Undergraduate",),
        ("Electrical Engineering", "Energy Engineering", "Mechanical Engineering"),
        "https://ocw.tudelft.nl/courses/solar-energy/",
        DELFT_LICENSE,
        VERIFICATION_DATE,
    ),
)


_SYNONYMS: dict[str, tuple[str, ...]] = {
    "statics": ("mechanics", "forces", "equilibrium", "structures"),
    "dynamics": ("mechanics", "motion", "control"),
    "circuit": ("circuits", "electrical", "network", "voltage", "current"),
    "circuits": ("electrical", "network", "voltage", "current"),
    "thermo": ("thermodynamics", "thermal", "heat", "energy"),
    "thermodynamics": ("thermal", "heat", "energy", "enthalpy"),
    "calculus": ("derivatives", "integrals", "rates", "optimization"),
    "fluids": ("fluid", "flow", "pressure", "control", "volume"),
    "probability": ("statistics", "random", "machine", "learning"),
    "aerospace": ("aerodynamics", "flight", "structures", "dynamics"),
}


def _tokens(value: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", value.lower()))
    expanded = set(words)
    for word in words:
        expanded.update(_SYNONYMS.get(word, ()))
    return expanded


def valid_resource_metadata(resource: LearningResource) -> bool:
    parsed = urlparse(resource.url)
    return bool(
        resource.institution.strip()
        and resource.title.strip()
        and resource.topics
        and resource.academic_levels
        and resource.engineering_majors
        and resource.license_note.strip()
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", resource.verification_date)
        and parsed.scheme == "https"
        and parsed.netloc
    )


def recommend_resources(
    query: str,
    profile: StudentProfile,
    *,
    limit: int = 4,
    catalog: tuple[LearningResource, ...] = RESOURCES,
) -> list[ResourceRecommendation]:
    """Rank verified records with deterministic, explainable token matching."""

    query_tokens = _tokens(query)
    goal_tokens = _tokens(profile.learning_goal)
    major = profile.engineering_major.lower()
    level_tokens = _tokens(profile.academic_level)
    ranked: list[ResourceRecommendation] = []

    for resource in catalog:
        if not resource.verified or not valid_resource_metadata(resource):
            continue
        topic_tokens = _tokens(" ".join(resource.topics))
        title_tokens = _tokens(resource.title)
        searchable = topic_tokens | title_tokens
        reasons: list[str] = []
        score = 0

        topic_matches = sorted(query_tokens & searchable)
        if topic_matches:
            score += min(30, 8 * len(topic_matches))
            reasons.append("topic match: " + ", ".join(topic_matches[:3]))

        compatible_majors = {item.lower() for item in resource.engineering_majors}
        if major in compatible_majors or "general engineering" in compatible_majors:
            score += 7
            reasons.append(f"fits {profile.engineering_major}")

        resource_levels = _tokens(" ".join(resource.academic_levels))
        if level_tokens & resource_levels:
            score += 4
            reasons.append(f"fits {profile.academic_level} level")

        goal_matches = sorted(goal_tokens & searchable)
        if goal_matches:
            score += min(6, 2 * len(goal_matches))
            reasons.append("supports learning goal")

        if score > 0:
            ranked.append(ResourceRecommendation(resource, score, tuple(reasons)))

    ranked.sort(key=lambda item: (-item.score, item.resource.institution, item.resource.title))
    return ranked[: max(0, limit)]


def format_recommendations(recommendations: list[ResourceRecommendation]) -> str:
    if not recommendations:
        return "No verified catalog resource matched this topic and profile."
    lines = ["Verified public university resources:"]
    for index, recommendation in enumerate(recommendations, 1):
        resource = recommendation.resource
        reason = "; ".join(recommendation.reasons)
        lines.extend(
            (
                f"{index}. {resource.title} — {resource.institution}",
                f"   {resource.url}",
                f"   Why: {reason}. Verified {resource.verification_date}.",
            )
        )
    return "\n".join(lines)

