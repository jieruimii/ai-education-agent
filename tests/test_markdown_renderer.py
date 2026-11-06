import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from markdown_renderer import parse_markdown, visible_text  # noqa: E402


class MarkdownRendererTests(unittest.TestCase):
    def test_headings_bold_lists_and_math_are_structured(self):
        source = (
            "### Newton's Second Law\n\n"
            "**For:** Undergraduate\n\n"
            "$$\\sum \\mathbf{F} = m\\mathbf{a}$$\n\n"
            "- $m$ is mass\n- $\\mathbf{a}$ is acceleration"
        )
        tokens = parse_markdown(source)
        self.assertTrue(any(t.style == "heading3" for t in tokens))
        self.assertTrue(any(t.style == "bold" and t.text == "For:" for t in tokens))
        self.assertTrue(any(t.kind == "math_display" for t in tokens))
        self.assertEqual(sum(t.style == "list_marker" for t in tokens), 2)
        rendered = visible_text(tokens)
        self.assertNotIn("###", rendered)
        self.assertNotIn("**", rendered)
        self.assertNotIn("$$", rendered)

    def test_inline_math_bracket_math_and_code_fences(self):
        source = (
            "Velocity is $v = dx/dt$ and \\(a=dv/dt\\).\n\n"
            "\\[E = mc^2\\]\n\n```python\nforce = mass * acceleration\n```"
        )
        tokens = parse_markdown(source)
        self.assertEqual(sum(t.kind == "math_inline" for t in tokens), 2)
        self.assertEqual(sum(t.kind == "math_display" for t in tokens), 1)
        self.assertTrue(any(t.style == "code_block" for t in tokens))

    def test_malformed_or_streaming_markers_remain_readable(self):
        source = "A partial response with **bold and $x + y"
        rendered = visible_text(parse_markdown(source))
        self.assertIn("partial response", rendered)
        self.assertIn("x + y", rendered)

    def test_markdown_links_are_inert_readable_text(self):
        tokens = parse_markdown("Read [MIT OCW](https://ocw.mit.edu/).")
        self.assertIn("MIT OCW (https://ocw.mit.edu/)", visible_text(tokens))
        self.assertTrue(any(t.style == "link" for t in tokens))

    def test_currency_is_not_misread_as_inline_math(self):
        tokens = parse_markdown("Pricing is $4 per input unit and $20 per output unit.")
        self.assertFalse(any(t.kind.startswith("math_") for t in tokens))
        self.assertIn("$4", visible_text(tokens))
        self.assertIn("$20", visible_text(tokens))


if __name__ == "__main__":
    unittest.main()
