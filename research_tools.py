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


SUBJECT_TERMS = {
    "physics": ("newton", "force", "motion", "velocity", "energy", "quantum", "relativity"),
    "mathematics": ("equation", "theorem", "integral", "derivative", "algebra", "geometry", "probability"),
    "chemistry": ("atom", "molecule", "reaction", "chemical", "acid", "base", "periodic"),
    "computer science": ("algorithm", "program", "software", "machine learning", "supervised", "unsupervised", "database"),
    "literature": ("novel", "poem", "author", "literature", "shakespeare", "metaphor", "character"),
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
        query_terms = {
            term for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2 and term not in {"explain", "what", "the", "and", "for"}
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
                minimum_overlap = 2 if len(query_terms) >= 3 else 1
                if overlap >= minimum_overlap:
                    evidence.append((overlap, WebEvidence(title, url, excerpt)))
        evidence.sort(key=lambda item: (-item[0], item[1].title))
        return tuple(item for _, item in evidence[: self.max_results])

