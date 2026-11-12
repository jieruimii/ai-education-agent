import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from education_harness import (  # noqa: E402
    REQUIRED_SECTIONS,
    HUMANITIES_SECTIONS,
    TutorAnswerHarness,
    build_question_contract,
    evaluate_answer,
    parse_critic,
)
from knowledge_base import find_concept_cards  # noqa: E402
from learning_core import MockEducationService, StudentProfile  # noqa: E402
from research_tools import (  # noqa: E402
    WebEvidence,
    WikipediaResearchTool,
    classify_subject,
    should_research,
)


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
        if isinstance(response, Exception):
            raise response
        yield response


class FakeResearcher:
    def __init__(self):
        self.calls = []

    def search(self, question, subject):
        self.calls.append((question, subject))
        return (
            WebEvidence(
                title="Newton's laws of motion",
                url="https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion",
                excerpt="The first law describes inertia and constant velocity when net force is zero.",
                relevance_score=95,
            ),
        )


class StubWikipediaResearchTool(WikipediaResearchTool):
    def _get(self, params):
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [
                        {"title": "Causes of the French Revolution"},
                        {"title": "French Revolution of 1848"},
                    ]
                }
            }
        return {
            "query": {
                "pages": {
                    "1": {
                        "title": "Causes of the French Revolution",
                        "fullurl": "https://en.wikipedia.org/wiki/Causes_of_the_French_Revolution",
                        "extract": "Multiple social, political, and financial causes contributed.",
                    },
                    "2": {
                        "title": "French Revolution of 1848",
                        "fullurl": "https://en.wikipedia.org/wiki/French_Revolution_of_1848",
                        "extract": "A different revolution occurred in 1848.",
                    },
                }
            }
        }


class HarnessContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = StudentProfile(
            engineering_major="Mechanical Engineering",
            academic_level="Undergraduate",
            active_course="MATH 407 • Probability Theory",
        )
        self.question = "Explain the throttling process"

    def test_throttling_retrieval_is_exact_and_source_approved(self):
        cards = find_concept_cards(self.question)
        self.assertEqual([card.card_id for card in cards], ["thermo.throttling.v1"])
        self.assertIn("ocw.mit.edu", cards[0].source_url)
        self.assertFalse(find_concept_cards("Explain beam bending", "Statics"))

    def test_contract_requires_holistic_sections(self):
        contract = build_question_contract(self.profile, self.question)
        self.assertEqual(contract.required_sections, REQUIRED_SECTIONS)
        self.assertEqual(contract.intent, "explain concept")
        self.assertFalse(contract.needs_uncertainty_note)

    def test_deterministic_rubric_rejects_generic_answer(self):
        cards = find_concept_cards(self.question)
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

    def test_legacy_course_cannot_override_machine_learning_question(self):
        question = (
            "Explain what is the difference between unsupervised machine learning "
            "and supervised machine learning"
        )
        cards = find_concept_cards(question)
        self.assertEqual(cards[0].card_id, "ml.supervised-vs-unsupervised.v1")
        contract = build_question_contract(self.profile, question, cards)
        self.assertIn("machine learning", contract.topic.lower())
        answer = MockEducationService().tutor_reply(self.profile, question).text
        self.assertIn("target label", answer.lower())
        self.assertIn("unsupervised clustering", answer.lower())
        self.assertNotIn("both pass with probability", answer.lower())

    def test_newton_question_routes_to_physics_and_holistic_mock_answer(self):
        question = "Explain what is Newton's 1st law"
        cards = find_concept_cards(question)
        contract = build_question_contract(self.profile, question, cards)
        self.assertEqual(contract.subject, "physics")
        self.assertEqual(cards[0].card_id, "physics.newton-first-law.v1")
        answer = MockEducationService().tutor_reply(self.profile, question).text
        for section in REQUIRED_SECTIONS:
            self.assertIn(section, answer)
        self.assertIn("net external force", answer.lower())
        self.assertIn("inertial reference frame", answer.lower())
        self.assertIn("openstax.org", answer.lower())

    def test_research_tool_events_are_observable_without_private_reasoning(self):
        question = "Search online and explain what is Newton's 1st law"
        draft = MockEducationService().tutor_reply(self.profile, question).text + (
            "\n\nThis conclusion applies to both an object remaining at rest and an object "
            "moving in a straight line at constant speed because both have zero acceleration."
        )
        transport = ScriptedTransport((draft, '{"pass": true, "feedback": ""}'))
        events = []
        final, trace = TutorAnswerHarness(
            transport, researcher=FakeResearcher()
        ).run(self.profile, question, progress=events.append)
        self.assertEqual(final, draft)
        self.assertEqual(trace.subject, "physics")
        self.assertIn("https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion", trace.source_urls)
        self.assertTrue(any("Classified subject: physics" in event for event in events))
        self.assertTrue(any("retrieved 1 source" in event for event in events))
        self.assertTrue(any("Release gate" in event for event in events))

    def test_reviewed_evidence_skips_unnecessary_web_research(self):
        draft = holistic_throttle_answer()
        transport = ScriptedTransport((draft, '{"pass": true, "feedback": ""}'))
        researcher = FakeResearcher()
        final, trace = TutorAnswerHarness(
            transport, researcher=researcher
        ).run(self.profile, self.question)
        self.assertEqual(final, draft)
        self.assertFalse(researcher.calls)
        self.assertTrue(any("skipped" in event for event in trace.tool_events))
        self.assertEqual(trace.quality_scores["tool precision"], 100)

    def test_empty_critic_retries_once_then_uses_deterministic_gate(self):
        draft = holistic_throttle_answer()
        no_text = RuntimeError("Meta completed without visible text")
        transport = ScriptedTransport((draft, no_text, no_text))
        events = []
        final, trace = TutorAnswerHarness(transport).run(
            self.profile, self.question, progress=events.append
        )
        self.assertEqual(final, draft)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(transport.calls[1]["max_tokens"], 900)
        self.assertEqual(transport.calls[2]["max_tokens"], 1200)
        self.assertTrue(trace.critic_retried)
        self.assertTrue(trace.deterministic_critic_fallback)
        self.assertTrue(trace.released)
        self.assertEqual(trace.quality_scores["reliability"], 75)
        self.assertTrue(any("deterministic rubric" in event for event in events))

    def test_empty_critic_retry_can_recover(self):
        draft = holistic_throttle_answer()
        transport = ScriptedTransport((
            draft,
            RuntimeError("Meta completed without visible text"),
            '{"pass": true, "feedback": ""}',
        ))
        final, trace = TutorAnswerHarness(transport).run(self.profile, self.question)
        self.assertEqual(final, draft)
        self.assertTrue(trace.critic_retried)
        self.assertFalse(trace.deterministic_critic_fallback)
        self.assertEqual(trace.quality_scores["reliability"], 85)

    def test_four_added_disciplines_have_reviewed_cards(self):
        cases = {
            "Evaluate the integral from 0 to 1 of x times exp(x squared)": "math.integration-by-substitution.v1",
            "Compare SN1 and SN2 substitution reactions": "chem.sn1-vs-sn2.v1",
            "Explain dramatic irony in Oedipus Rex": "literature.dramatic-irony-oedipus.v1",
            "Explain the main causes of the French Revolution": "history.french-revolution-causes.v1",
        }
        for question, card_id in cases.items():
            with self.subTest(question=question):
                self.assertEqual(find_concept_cards(question)[0].card_id, card_id)

    def test_word_budget_rejects_an_overlong_answer(self):
        cards = find_concept_cards(self.question)
        contract = build_question_contract(self.profile, self.question, cards)
        overlong = holistic_throttle_answer() + (" extra filler" * 500)
        report = evaluate_answer(self.question, overlong, contract, cards)
        self.assertFalse(report.passed)
        self.assertTrue(any("too long" in warning for warning in report.warnings))

    def test_release_gate_rejects_heading_before_direct_answer(self):
        cards = find_concept_cards("Compare SN1 and SN2 substitution reactions")
        contract = build_question_contract(
            self.profile, "Compare SN1 and SN2 substitution reactions", cards
        )
        answer = "## What it is\n" + ("SN1 and SN2 differ in mechanism and rate law. " * 35)
        answer += "\n" + "\n".join(
            f"## {section}\nSupporting explanation." for section in REQUIRED_SECTIONS[1:]
        )
        answer += f"\nSource: {cards[0].source_title} — {cards[0].source_url}"
        report = evaluate_answer(
            "Compare SN1 and SN2 substitution reactions", answer, contract, cards
        )
        self.assertFalse(report.passed)
        self.assertTrue(any("direct sentence" in warning for warning in report.warnings))

    def test_conditional_research_and_relevance_filtering(self):
        self.assertFalse(should_research(
            "Explain Newton's first law", has_reviewed_evidence=True
        ))
        self.assertTrue(should_research(
            "Find the latest Newton-law revision", has_reviewed_evidence=True
        ))
        self.assertTrue(should_research(
            "Explain an unmatched topic", has_reviewed_evidence=False
        ))
        evidence = StubWikipediaResearchTool().search(
            "French Revolution causes", "history"
        )
        self.assertEqual([item.title for item in evidence], [
            "Causes of the French Revolution"
        ])
        self.assertGreaterEqual(evidence[0].relevance_score, 60)

    def test_subject_classifier_covers_cross_disciplinary_questions(self):
        cases = {
            "Evaluate this integral": "mathematics",
            "Explain an acid-base reaction": "chemistry",
            "Analyze the metaphor in this poem": "literature",
            "Why did this empire collapse in history?": "history",
            "Compare supervised and unsupervised machine learning": "computer science",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_subject(question), expected)

    def test_humanities_contract_does_not_force_a_governing_equation(self):
        question = "Explain dramatic irony in Sophocles' Oedipus Rex"
        contract = build_question_contract(self.profile, question)
        self.assertEqual(contract.subject, "literature")
        self.assertEqual(contract.required_sections, HUMANITIES_SECTIONS)
        self.assertNotIn("Governing equation", contract.required_sections)

    def test_natural_machine_learning_phrase_matches_reviewed_card(self):
        cards = find_concept_cards(
            "Compare supervised and unsupervised machine learning"
        )
        self.assertEqual(cards[0].card_id, "ml.supervised-vs-unsupervised.v1")


if __name__ == "__main__":
    unittest.main()

