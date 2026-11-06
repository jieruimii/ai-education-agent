"""Small, offline Markdown/math parser used by the Tkinter chat renderer.

The application only needs a safe presentation layer, not HTML.  This module
therefore turns the subset commonly produced by tutoring models into neutral
tokens that a ``tk.Text`` widget can style.  It never executes HTML or follows
links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RenderToken:
    kind: str
    text: str
    style: str = "body"


_BLOCK_PATTERN = re.compile(
    r"(?P<fence>```(?:[^\n`]*)\n.*?```)|"
    r"(?P<dollar>\$\$.*?\$\$)|"
    r"(?P<bracket>\\\[.*?\\\])",
    re.DOTALL,
)
_INLINE_PATTERN = re.compile(
    r"(?P<code>`[^`\n]+`)|"
    r"(?P<bold>\*\*.+?\*\*)|"
    r"(?P<math>\$(?!\$).*?(?<!\$)\$|\\\(.+?\\\))|"
    r"(?P<italic>(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*))|"
    r"(?P<link>\[[^\]]+\]\([^)]+\))"
)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_UNORDERED = re.compile(r"^\s*[-*+]\s+(.+)$")
_ORDERED = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")


def _inline_tokens(text: str, base_style: str) -> list[RenderToken]:
    tokens: list[RenderToken] = []
    cursor = 0
    for match in _INLINE_PATTERN.finditer(text):
        if match.start() > cursor:
            tokens.append(RenderToken("text", text[cursor:match.start()], base_style))
        raw = match.group(0)
        if match.lastgroup == "code":
            tokens.append(RenderToken("text", raw[1:-1], "code"))
        elif match.lastgroup == "bold":
            tokens.append(RenderToken("text", raw[2:-2], "bold"))
        elif match.lastgroup == "italic":
            tokens.append(RenderToken("text", raw[1:-1], "italic"))
        elif match.lastgroup == "math":
            # Do not interpret a price range such as "$4 ... $20" as math.
            next_character = text[match.end():match.end() + 1]
            if raw.startswith("$") and raw[1:2].isdigit() and next_character.isdigit():
                tokens.append(RenderToken("text", raw, base_style))
            else:
                expression = raw[2:-2] if raw.startswith(r"\(") else raw[1:-1]
                tokens.append(RenderToken("math_inline", expression.strip(), "math"))
        else:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", raw).groups()
            tokens.append(RenderToken("text", f"{label} ({url})", "link"))
        cursor = match.end()
    if cursor < len(text):
        tokens.append(RenderToken("text", text[cursor:], base_style))
    return tokens


def _plain_block(text: str) -> list[RenderToken]:
    tokens: list[RenderToken] = []
    lines = text.splitlines()
    for line in lines:
        heading = _HEADING.match(line)
        unordered = _UNORDERED.match(line)
        ordered = _ORDERED.match(line)
        if heading:
            level = min(len(heading.group(1)), 3)
            tokens.extend(_inline_tokens(heading.group(2), f"heading{level}"))
        elif unordered:
            tokens.append(RenderToken("text", "•  ", "list_marker"))
            tokens.extend(_inline_tokens(unordered.group(1), "body"))
        elif ordered:
            tokens.append(RenderToken("text", f"{ordered.group(1)}.  ", "list_marker"))
            tokens.extend(_inline_tokens(ordered.group(2), "body"))
        elif line.startswith(">"):
            tokens.extend(_inline_tokens(line[1:].lstrip(), "quote"))
        else:
            tokens.extend(_inline_tokens(line, "body"))
        tokens.append(RenderToken("newline", "\n"))
    return tokens


def parse_markdown(text: str) -> list[RenderToken]:
    """Parse model-authored Markdown into safe, display-only tokens."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[RenderToken] = []
    cursor = 0
    for match in _BLOCK_PATTERN.finditer(normalized):
        tokens.extend(_plain_block(normalized[cursor:match.start()]))
        raw = match.group(0)
        if match.lastgroup == "fence":
            first_newline = raw.find("\n")
            code = raw[first_newline + 1:-3] if first_newline >= 0 else raw[3:-3]
            tokens.append(RenderToken("text", code.rstrip("\n"), "code_block"))
        else:
            expression = raw[2:-2].strip()
            tokens.append(RenderToken("math_display", expression, "math"))
        tokens.append(RenderToken("newline", "\n"))
        cursor = match.end()
    tokens.extend(_plain_block(normalized[cursor:]))
    # Avoid accumulating extra blank lines caused by block boundaries.
    compacted: list[RenderToken] = []
    for token in tokens:
        if (
            token.kind == "newline"
            and len(compacted) >= 2
            and compacted[-1].kind == compacted[-2].kind == "newline"
        ):
            continue
        compacted.append(token)
    return compacted


def visible_text(tokens: list[RenderToken]) -> str:
    """Return a plain-text representation for tests and accessibility checks."""
    parts: list[str] = []
    for token in tokens:
        if token.kind.startswith("math_"):
            parts.append(token.text)
        else:
            parts.append(token.text)
    return "".join(parts)
