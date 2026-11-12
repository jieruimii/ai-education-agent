"""Bounded online research tools for the education-agent harness.

The tool uses only Wikipedia's public MediaWiki API. It cannot fetch arbitrary
URLs, execute page code, or follow redirects supplied by a model. Retrieved
text is evidence, never instructions.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class WebEvidence:
    title: str
    url: str
    excerpt: str
    source: str = "Wikipedia"
    relevance_score: int = 0


SUBJECT_TERMS = {
    "physics": ("newton", "force", "motion", "velocity", "energy", "quantum", "relativity"),
    "mathematics": ("equation", "theorem", "integral", "derivative", "algebra", "geometry", "probability"),
    "chemistry": ("atom", "molecule", "reaction", "chemical", "acid", "base", "periodic"),
    "computer science": ("algorithm", "program", "software", "machine learning", "supervised", "unsupervised", "database"),
    "literature": (
        "novel", "poem", "author", "literature", "shakespeare", "metaphor",
        "character", "dramatic irony", "drama", "play", "sophocles", "oedipus",
    ),
    "history": ("history", "war", "revolution", "empire", "dynasty", "century", "president"),
    "engineering": ("circuit", "beam", "thermodynamic", "throttl", "control system", "mechanical", "electrical"),
}


def classify_subject(question: str) -> str:
    text = question.lower()
    scores = {
        subject: sum(1 for term in terms if term in text)
        for subject, terms in SUBJECT_TERMS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "general academic"


def should_research(question: str, *, has_reviewed_evidence: bool) -> bool:
    """Use the web only when local evidence is absent or freshness is explicit."""

    if not has_reviewed_evidence:
        return True
    lowered = question.lower()
    freshness_terms = (
        "latest", "current", "today", "recent", "as of", "this year",
        "search online", "look up", "browse", "web source",
    )
    return any(term in lowered for term in freshness_terms)


class WikipediaResearchTool:
    """Search and retrieve short encyclopedia evidence from a fixed endpoint."""

    endpoint = "https://en.wikipedia.org/w/api.php"

    def __init__(self, *, timeout: float = 8.0, max_results: int = 3) -> None:
        self.timeout = timeout
        self.max_results = max(1, min(max_results, 5))

    def _get(self, params: dict[str, str]) -> dict:
        url = self.endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AI-Education-Agent/0.2 (academic prototype)"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read(1_000_000).decode("utf-8"))

    def search(self, question: str, subject: str) -> tuple[WebEvidence, ...]:
        query = " ".join(question.split())[:300]
        lowered = query.lower()
        # Normalize a few common academic phrasings to encyclopedia article
        # titles. The original question remains in the generation prompt.
        if "buffer" in lowered and any(term in lowered for term in ("acid", "base", "ph")):
            query = "buffer solution Henderson Hasselbalch"
        elif "french" in lowered and "revolution" in lowered:
            query = "French Revolution causes"
        elif "dramatic irony" in lowered and "oedipus" in lowered:
            query = "dramatic irony Sophocles Oedipus Rex"
        query_terms = {
            term for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in {
                "analyze", "compare", "directly", "evaluate", "explain",
                "give", "main", "show", "state", "the", "and", "for", "what",
            }
        }
        search = self._get({
            "action": "query",
            "list": "search",
            "srsearch": f"{query} {subject}",
            "srlimit": str(self.max_results),
            "format": "json",
            "utf8": "1",
        })
        titles = [item["title"] for item in search.get("query", {}).get("search", [])]
        if not titles:
            return ()
        pages = self._get({
            "action": "query",
            "prop": "extracts|info",
            "titles": "|".join(titles),
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "format": "json",
            "utf8": "1",
        }).get("query", {}).get("pages", {})
        evidence = []
        for page in pages.values():
            excerpt = re.sub(r"\s+", " ", page.get("extract", "")).strip()[:2400]
            url = page.get("fullurl", "")
            if excerpt and url.startswith("https://en.wikipedia.org/"):
                title = page.get("title", "Wikipedia")
                title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
                overlap = len(query_terms & title_terms)
                coverage = overlap / max(1, len(query_terms))
                score = round(100 * coverage)
                # A dated event is usually a narrower, different topic when the
                # learner did not ask for that year (for example, 1848 vs. 1789).
                title_years = set(re.findall(r"\b(?:1[5-9]|20)\d{2}\b", title))
                query_years = set(re.findall(r"\b(?:1[5-9]|20)\d{2}\b", query))
                if title_years and not title_years.intersection(query_years):
                    score -= 35
                if score >= 60:
                    evidence.append((score, WebEvidence(
                        title, url, excerpt, relevance_score=max(0, score)
                    )))
        evidence.sort(key=lambda item: (-item[0], item[1].title))
        return tuple(item for _, item in evidence[: self.max_results])

