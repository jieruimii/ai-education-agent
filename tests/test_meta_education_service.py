import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning_core import (  # noqa: E402
    EducationTurn,
    ProblemSolvingSession,
    StudentProfile,
    WorkflowStage,
)
from meta_education_service import (  # noqa: E402
    MetaEducationService,
    filter_provider_urls,
    select_provider,
)
from main import EducationAgentApp, MetaModelService  # noqa: E402
from resource_catalog import RESOURCES, recommend_resources  # noqa: E402


class FakeTransport:
    def __init__(self, chunks=("A clear explanation.",), error=None):
        self.chunks = chunks
        self.error = error
        self.calls = []

    def stream_reply(self, messages, *, max_tokens):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        yield from self.chunks


class FakeEvent:
    type = "response.output_text.delta"
    delta = "provider text"


class FakeStream:
    def __init__(self, recorder, request):
        self.recorder = recorder
        self.request = request

    def __enter__(self):
        self.recorder.append(self.request)
        return iter((FakeEvent(),))

    def __exit__(self, *_args):
        return False


class FakeResponses:
    def __init__(self):
        self.requests = []

    def stream(self, **request):
        return FakeStream(self.requests, request)


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class ProviderSelectionTests(unittest.TestCase):
    def test_mock_never_requires_a_key(self):
        self.assertEqual(select_provider("mock", "").name, "mock")

    def test_auto_uses_meta_only_with_a_non_placeholder_key(self):
        self.assertEqual(select_provider("auto", "").name, "mock")
        self.assertEqual(select_provider("auto", "replace_with_key").name, "mock")
        self.assertEqual(select_provider("auto", "configured-secret").name, "meta")

    def test_explicit_meta_without_key_fails_safely(self):
        selection = select_provider("meta", "")
        self.assertEqual(selection.name, "mock")
        self.assertIn("MODEL_API_KEY", selection.error)
        self.assertNotIn("Bearer", selection.error)

    def test_invalid_mode_falls_back_to_mock(self):
        selection = select_provider("surprise", "configured-secret")
        self.assertEqual(selection.name, "mock")
        self.assertIn("AI_AGENT_MODE", selection.error)


class MetaEducationServiceTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport(("First chunk. ", "Second chunk."))
        self.service = MetaEducationService(self.transport, max_tokens=777)
        self.profile = StudentProfile(
            engineering_major="Electrical Engineering",
            academic_level="Graduate",
            learning_goal="Master feedback control",
            active_course="EE 501 • Linear Systems",
            explanation_preference="Concise",
        )

    def test_tutor_request_construction_and_streaming(self):
        turn = self.service.tutor_reply(self.profile, "Explain stability")
        self.assertEqual(turn.text, "First chunk. Second chunk.")
        self.assertFalse(turn.mock)
        call = self.transport.calls[0]
        self.assertEqual(call["max_tokens"], 777)
        self.assertEqual(call["messages"][0]["role"], "developer")
        self.assertEqual(call["messages"][-1]["role"], "user")

    def test_profile_context_excludes_legacy_active_course(self):
        list(self.service.stream_tutor_reply(self.profile, "Explain poles"))
        request_text = "\n".join(item["content"] for item in self.transport.calls[0]["messages"])
        self.assertIn("Electrical Engineering", request_text)
        self.assertIn("Graduate", request_text)
        self.assertIn("Master feedback control", request_text)
        self.assertNotIn("EE 501", request_text)
        self.assertNotIn("Active course", request_text)
        self.assertIn("Concise", request_text)

    def test_history_is_bounded_and_current_question_is_last(self):
        history = [("user", f"question {i}") for i in range(10)]
        messages = self.service.build_tutor_messages(
            self.profile, "current question", history
        )
        self.assertEqual(len(messages), 8)  # developer + six history + current
        self.assertIn("question 4", messages[1]["content"])
        self.assertIn("current question", messages[-1]["content"])

    def test_provider_history_excludes_local_filenames_and_catalog_appendix(self):
        history = [
            ("user", "Explain this excerpt\n📎 private-coursework.pdf"),
            (
                "ai",
                "Explanation text.\n\nVerified public university resources:\n"
                "1. Example\n   https://catalog.example/item",
            ),
        ]
        messages = self.service.build_tutor_messages(
            self.profile, "Continue", history
        )
        combined = "\n".join(item["content"] for item in messages)
        self.assertNotIn("private-coursework.pdf", combined)
        self.assertNotIn("catalog.example", combined)
        self.assertIn("Explanation text.", combined)

    def test_python_state_machine_owns_problem_transition(self):
        session = ProblemSolvingSession()
        directive = self.service.prepare_problem_turn(
            self.profile,
            session,
            "Given a voltage, I need to find current; I tried Ohm's law.",
        )
        self.assertEqual(session.stage, WorkflowStage.DESIGN_HINT)
        stage_before_generation = session.stage
        list(
            self.service.stream_problem_language(
                self.profile, session, "hint please", directive
            )
        )
        self.assertEqual(session.stage, stage_before_generation)
        request_text = "\n".join(
            item["content"] for item in self.transport.calls[-1]["messages"]
        )
        self.assertIn("PYTHON-OWNED WORKFLOW STAGE: Design Hint", request_text)

    def test_direct_answer_refusal_does_not_call_provider(self):
        session = ProblemSolvingSession()
        turn = self.service.problem_reply(
            self.profile, session, "Give me the final answer"
        )
        self.assertTrue(turn.flagged)
        self.assertEqual(session.stage, WorkflowStage.DIAGNOSE)
        self.assertFalse(self.transport.calls)

    def test_provider_leakage_is_replaced_with_python_directive(self):
        leaky = MetaEducationService(FakeTransport(("The final answer is 42.",)))
        session = ProblemSolvingSession()
        turn = leaky.problem_reply(self.profile, session, "A beam problem")
        self.assertTrue(turn.flagged)
        self.assertNotIn("42", turn.text)
        self.assertIn("Stage 1", turn.text)

    def test_timeout_is_propagated_for_ui_safe_mapping(self):
        service = MetaEducationService(
            FakeTransport(error=TimeoutError("secret-bearing raw body"))
        )
        with self.assertRaises(TimeoutError):
            service.tutor_reply(self.profile, "Explain damping")

    def test_prompts_do_not_contain_or_request_recommendation_urls(self):
        messages = self.service.build_tutor_messages(self.profile, "Explain damping")
        combined = "\n".join(item["content"] for item in messages)
        self.assertNotIn("http://", combined)
        self.assertNotIn("https://", combined)
        self.assertIn("Do not invent citations, links", combined)

    def test_only_catalog_urls_survive_provider_filter(self):
        allowed = {RESOURCES[0].url}
        text = f"Use {RESOURCES[0].url} and https://invented.example/course."
        filtered, removed = filter_provider_urls(text, allowed)
        self.assertTrue(removed)
        self.assertIn(RESOURCES[0].url, filtered)
        self.assertNotIn("invented.example", filtered)

    def test_problem_sessions_remain_isolated(self):
        first = ProblemSolvingSession()
        second = ProblemSolvingSession()
        self.service.prepare_problem_turn(
            self.profile,
            first,
            "Given a load, find the reaction; I tried a force balance.",
        )
        self.assertEqual(first.stage, WorkflowStage.DESIGN_HINT)
        self.assertEqual(second.stage, WorkflowStage.DIAGNOSE)
        self.assertFalse(second.known_information)

    def test_recommendations_remain_local_catalog_records(self):
        results = recommend_resources("linear systems", self.profile, limit=3)
        catalog_urls = {item.url for item in RESOURCES}
        self.assertTrue(results)
        self.assertTrue(all(item.resource.url in catalog_urls for item in results))

    def test_finalize_empty_output_uses_deterministic_fallback(self):
        directive = EducationTurn(
            "Stage 2 — Design Hint. Draw the system boundary.",
            stage=WorkflowStage.DESIGN_HINT,
        )
        finalized = self.service.finalize_problem_text("", directive)
        self.assertEqual(finalized.text, directive.text)
        self.assertEqual(finalized.stage, WorkflowStage.DESIGN_HINT)


class MetaTransportTests(unittest.TestCase):
    def test_verified_responses_api_request_shape(self):
        transport = MetaModelService.__new__(MetaModelService)
        transport.client = FakeClient()
        transport.model = "muse-wonder-1.0"
        chunks = list(
            transport.stream_reply(
                [{"role": "user", "content": "Explain feedback"}],
                max_tokens=512,
            )
        )
        self.assertEqual(chunks, ["provider text"])
        request = transport.client.responses.requests[0]
        self.assertEqual(request["model"], "muse-wonder-1.0")
        self.assertEqual(request["max_output_tokens"], 512)
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["input"][0]["role"], "user")

    def test_timeout_message_does_not_expose_exception_text(self):
        safe = EducationAgentApp._safe_provider_error(
            TimeoutError("MODEL_API_KEY=must-not-appear")
        )
        self.assertIn("could not be reached", safe)
        self.assertNotIn("must-not-appear", safe)


if __name__ == "__main__":
    unittest.main()
