import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from education_harness import (  # noqa: E402
    REQUIRED_SECTIONS,
    TutorAnswerHarness,
    build_question_contract,
    evaluate_answer,
    parse_critic,
)
from knowledge_base import find_concept_cards  # noqa: E402
from learning_core import MockEducationService, StudentProfile  # noqa: E402


def holistic_throttle_answer() -> str:
    body = (
        "A throttling valve creates a pressure drop while specific enthalpy remains approximately "
        "constant under the usual steady-flow assumptions. The restriction causes irreversible "
        "dissipation, so an adiabatic throttle increases entropy. "
    )
    sections = {
        "What it is": body,
        "How it works": body * 2,
        "Governing equation": (
            "The steady energy balance reduces to h_1 = h_2 when the device is adiabatic, "
            "has no shaft work, and has negligible kinetic and potential energy changes."
        ),
        "Assumptions and limits": (
            "Assume steady flow, no heat transfer, no work, and negligible velocity and elevation changes."
        ),
        "Concrete example": (
            "An expansion valve lowers refrigerant pressure and saturation temperature; some liquid flashes."
        ),
        "Common misconception": (
            "Isenthalpic does not always mean isothermal, and irreversible throttling is not isentropic."
        ),
        "Check your understanding": "Which discarded energy term would matter for a high outlet velocity?",
    }
    answer = body + "\n\n" + "\n\n".join(
        f"## {name}\n{text}" for name, text in sections.items()
    )
    answer += (
        "\n\nSource: MIT Unified Engineering Thermodynamics, Chapter 4 — "
        "https://ocw.mit.edu/ans7870/16/16.unified/thermoF03/chapter_4.htm"
    )
    return answer


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream_reply(self, messages, *, max_tokens):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        response = self.responses.pop(0)
        yield response


class HarnessContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = StudentProfile(
            engineering_major="Mechanical Engineering",
            academic_level="Undergraduate",
            active_course="Thermodynamics",
        )
        self.question = "Explain the throttling process"

    def test_throttling_retrieval_is_exact_and_source_approved(self):
        cards = find_concept_cards(self.question, self.profile.active_course)
        self.assertEqual([card.card_id for card in cards], ["thermo.throttling.v1"])
        self.assertIn("ocw.mit.edu", cards[0].source_url)
        self.assertFalse(find_concept_cards("Explain beam bending", "Statics"))

    def test_contract_requires_holistic_sections(self):
        contract = build_question_contract(self.profile, self.question)
        self.assertEqual(contract.required_sections, REQUIRED_SECTIONS)
        self.assertEqual(contract.intent, "explain concept")
        self.assertFalse(contract.needs_uncertainty_note)

    def test_deterministic_rubric_rejects_generic_answer(self):
        cards = find_concept_cards(self.question, self.profile.active_course)
        contract = build_question_contract(self.profile, self.question, cards)
        report = evaluate_answer(
            self.question,
            "Throttling is important and relates to other thermodynamics classes.",
            contract,
            cards,
        )
        self.assertFalse(report.passed)
        self.assertGreaterEqual(len(report.missing_sections), 6)

    def test_valid_draft_passes_critic_without_revision(self):
        draft = holistic_throttle_answer()
        transport = ScriptedTransport((draft, '{"pass": true, "feedback": ""}'))
        harness = TutorAnswerHarness(transport)
        final, trace = harness.run(self.profile, self.question)
        self.assertEqual(final, draft)
        self.assertEqual(len(transport.calls), 2)
        self.assertFalse(trace.revised)
        self.assertTrue(trace.released)
        self.assertEqual(trace.evidence_ids, ("thermo.throttling.v1",))

    def test_failed_draft_is_revised_before_release(self):
        draft = "Throttling is a general thermodynamics topic."
        revised = holistic_throttle_answer()
        transport = ScriptedTransport((
            draft,
            '{"pass": false, "feedback": "Explain the energy balance and irreversibility."}',
            revised,
        ))
        final, trace = TutorAnswerHarness(transport).run(self.profile, self.question)
        self.assertEqual(final, revised)
        self.assertTrue(trace.revised)
        self.assertIn("revise", trace.stages)
        self.assertGreater(trace.final_score, trace.initial_score)

    def test_invalid_critic_json_triggers_revision(self):
        passed, feedback = parse_critic("not-json")
        self.assertFalse(passed)
        self.assertIn("valid JSON", feedback)

    def test_mock_regression_answer_is_holistic_and_corrects_misconceptions(self):
        answer = MockEducationService().tutor_reply(self.profile, self.question).text
        for section in REQUIRED_SECTIONS:
            self.assertIn(section, answer)
        self.assertIn("h_1\\approx h_2", answer)
        self.assertIn("not necessarily isothermal", answer)
        self.assertIn("entropy increases", answer)


if __name__ == "__main__":
    unittest.main()

