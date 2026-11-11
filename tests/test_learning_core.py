import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning_core import (  # noqa: E402
    MockEducationService,
    ProblemSolvingSession,
    StudentProfile,
    WorkflowStage,
    contains_output_leakage,
    is_direct_answer_request,
    progressive_hint,
    safeguard_problem_solver_output,
    select_hint_level,
)
from resource_catalog import (  # noqa: E402
    LearningResource,
    RESOURCES,
    recommend_resources,
    valid_resource_metadata,
)


def diagnosed_session() -> ProblemSolvingSession:
    session = ProblemSolvingSession()
    service = MockEducationService()
    profile = StudentProfile()
    service.problem_reply(
        profile,
        session,
        "Given a 10 kN load, I need to find both reactions; I tried a free-body diagram.",
    )
    return session


class ProblemSolverStateTests(unittest.TestCase):
    def setUp(self):
        self.service = MockEducationService()
        self.profile = StudentProfile(
            engineering_major="Mechanical Engineering",
            academic_level="First-year Undergraduate",
            active_course="ENGR 101 • Statics",
        )

    def test_new_session_starts_at_diagnose(self):
        self.assertEqual(ProblemSolvingSession().stage, WorkflowStage.DIAGNOSE)

    def test_diagnosis_does_not_advance_without_evidence(self):
        session = ProblemSolvingSession()
        self.service.problem_reply(self.profile, session, "A beam problem")
        self.assertEqual(session.stage, WorkflowStage.DIAGNOSE)

    def test_diagnose_to_design_hint_requires_meaningful_evidence(self):
        session = diagnosed_session()
        self.assertEqual(session.stage, WorkflowStage.DESIGN_HINT)
        self.assertTrue(session.known_information)
        self.assertTrue(session.unknown_variables)

    def test_design_hint_does_not_advance_on_acknowledgement(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "okay")
        self.assertEqual(session.stage, WorkflowStage.DESIGN_HINT)

    def test_design_hint_to_gather_report_on_explicit_hint(self):
        session = diagnosed_session()
        turn = self.service.problem_reply(self.profile, session, "I am stuck; give me a hint")
        self.assertEqual(session.stage, WorkflowStage.GATHER_REPORT)
        self.assertIn("level", turn.text.lower())

    def test_gather_report_waits_for_substantive_attempt(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "hint please")
        self.service.problem_reply(self.profile, session, "okay")
        self.assertEqual(session.stage, WorkflowStage.GATHER_REPORT)

    def test_gather_report_to_iterate_on_attempt(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "hint please")
        self.service.problem_reply(
            self.profile,
            session,
            "I used moment balance about A because that removes one reaction force.",
        )
        self.assertEqual(session.stage, WorkflowStage.ITERATE)
        self.assertEqual(len(session.attempts), 1)

    def test_gather_report_stuck_returns_to_design_hint(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "hint please")
        self.service.problem_reply(self.profile, session, "I don't know")
        self.assertEqual(session.stage, WorkflowStage.DESIGN_HINT)

    def test_iterate_routes_insufficient_report_back_to_gather(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "hint please")
        self.service.problem_reply(
            self.profile,
            session,
            "I used moment balance about A because that removes one reaction force.",
        )
        self.service.problem_reply(self.profile, session, "okay")
        self.assertEqual(session.stage, WorkflowStage.GATHER_REPORT)

    def test_iterate_confusion_returns_to_design_hint(self):
        session = diagnosed_session()
        self.service.problem_reply(self.profile, session, "hint please")
        self.service.problem_reply(
            self.profile,
            session,
            "I used moment balance about A because that removes one reaction force.",
        )
        self.service.problem_reply(self.profile, session, "I am confused")
        self.assertEqual(session.stage, WorkflowStage.DESIGN_HINT)

    def test_progressive_hint_escalates_with_observed_struggle(self):
        session = ProblemSolvingSession(stuck_count=0)
        self.assertEqual(select_hint_level(session), 1)
        session.stuck_count = 1
        self.assertEqual(select_hint_level(session), 2)
        session.stuck_count = 5
        self.assertEqual(select_hint_level(session), 3)
        self.assertNotEqual(progressive_hint("statics", 1), progressive_hint("statics", 3))

    def test_direct_answer_request_is_flagged_and_blocked(self):
        session = ProblemSolvingSession()
        turn = self.service.problem_reply(self.profile, session, "Just solve it for me")
        self.assertTrue(turn.flagged)
        self.assertNotIn("final answer is", turn.text.lower())
        self.assertEqual(session.stage, WorkflowStage.DIAGNOSE)
        self.assertTrue(is_direct_answer_request("Give me the final answer"))

    def test_output_safeguard_replaces_known_leakage(self):
        unsafe = "The final answer is 42."
        self.assertTrue(contains_output_leakage(unsafe))
        safe, blocked = safeguard_problem_solver_output(unsafe)
        self.assertTrue(blocked)
        self.assertNotEqual(safe, unsafe)

    def test_sessions_are_isolated(self):
        first = diagnosed_session()
        second = ProblemSolvingSession()
        self.service.problem_reply(self.profile, first, "hint please")
        self.assertEqual(first.stage, WorkflowStage.GATHER_REPORT)
        self.assertEqual(second.stage, WorkflowStage.DIAGNOSE)
        self.assertFalse(second.hints_given)


class TutorPersonalizationTests(unittest.TestCase):
    def setUp(self):
        self.service = MockEducationService()

    def test_tutor_can_provide_complete_worked_solution(self):
        profile = StudentProfile(active_course="EE 201 • Circuits")
        turn = self.service.tutor_reply(profile, "Show a practice series-circuit example")
        self.assertIn("Worked practice example", turn.text)
        self.assertIn("I = 12/6 = 2 A", turn.text)
        self.assertIn("Comprehension check", turn.text)

    def test_profiles_change_same_topic_response(self):
        first_year = StudentProfile(
            engineering_major="Mechanical Engineering",
            academic_level="First-year Undergraduate",
            active_course="ENGR 101 • Statics",
            explanation_preference="Step-by-step",
        )
        graduate = StudentProfile(
            engineering_major="Electrical Engineering",
            academic_level="Graduate",
            active_course="ENGR 101 • Statics",
            explanation_preference="Concise",
        )
        question = "Explain reaction forces in a beam"
        first = self.service.tutor_reply(first_year, question).text
        second = self.service.tutor_reply(graduate, question).text
        self.assertNotEqual(first, second)
        self.assertIn("physical intuition", first)
        self.assertIn("model limits", second)

    def test_mock_service_does_not_require_api_key(self):
        previous = os.environ.pop("MODEL_API_KEY", None)
        try:
            service = MockEducationService()
            turn = service.tutor_reply(StudentProfile(), "Explain calculus")
            self.assertTrue(turn.mock)
            self.assertIn("deterministic local response", turn.text)
        finally:
            if previous is not None:
                os.environ["MODEL_API_KEY"] = previous

    def test_current_question_overrides_legacy_active_course(self):
        profile = StudentProfile(active_course="MATH 407 • Probability Theory")
        question = "Compare supervised and unsupervised machine learning"
        answer = self.service.tutor_reply(profile, question).text
        self.assertIn("target label", answer.lower())
        self.assertNotIn("two components both pass", answer.lower())

    def test_newton_resources_do_not_fall_back_to_calculus(self):
        profile = StudentProfile(
            engineering_major="General Engineering",
            active_course="MATH 407 • Probability Theory",
        )
        results = recommend_resources("Explain Newton's first law", profile, limit=3)
        titles = [item.resource.title for item in results]
        self.assertTrue(titles)
        self.assertEqual(titles[0], "Engineering Mechanics")
        self.assertFalse(any("Calculus" in title for title in titles))


class ResourceCatalogTests(unittest.TestCase):
    def test_catalog_size_and_metadata(self):
        self.assertGreaterEqual(len(RESOURCES), 10)
        self.assertLessEqual(len(RESOURCES), 15)
        for resource in RESOURCES:
            self.assertTrue(resource.verified)
            self.assertTrue(valid_resource_metadata(resource), resource.title)

    def test_unverified_records_are_excluded(self):
        unverified = LearningResource(
            institution="Example University",
            title="Unverified Mechanics",
            topics=("mechanics",),
            academic_levels=("Undergraduate",),
            engineering_majors=("Mechanical Engineering",),
            url="https://example.edu/mechanics",
            license_note="Unknown",
            verification_date="2026-11-05",
            verified=False,
        )
        profile = StudentProfile(engineering_major="Mechanical Engineering")
        results = recommend_resources("mechanics", profile, catalog=RESOURCES + (unverified,))
        self.assertNotIn("Unverified Mechanics", [item.resource.title for item in results])

    def test_rankings_change_by_profile_for_same_topic(self):
        mechanical = StudentProfile(
            engineering_major="Mechanical Engineering",
            academic_level="First-year Undergraduate",
            active_course="ENGR 101 • Statics",
            learning_goal="Understand forces and structures",
        )
        electrical = StudentProfile(
            engineering_major="Electrical Engineering",
            academic_level="Graduate",
            active_course="EE 501 • Linear Systems",
            learning_goal="Apply control and circuits",
        )
        query = "dynamics and system modeling"
        mechanical_titles = [item.resource.title for item in recommend_resources(query, mechanical, limit=5)]
        electrical_titles = [item.resource.title for item in recommend_resources(query, electrical, limit=5)]
        self.assertNotEqual(mechanical_titles, electrical_titles)
        self.assertIn("2.003SC Engineering Dynamics", mechanical_titles)
        self.assertIn("EE263 Introduction to Linear Dynamical Systems", electrical_titles)

    def test_recommendations_are_explainable(self):
        profile = StudentProfile(
            engineering_major="Electrical Engineering",
            academic_level="Undergraduate",
            active_course="EE 201 • Circuits",
        )
        results = recommend_resources("series circuit current and voltage", profile, limit=3)
        self.assertTrue(results)
        for item in results:
            self.assertGreater(item.score, 0)
            self.assertTrue(item.reasons)

    def test_current_query_controls_resources_not_legacy_course(self):
        profile = StudentProfile(
            engineering_major="Data Science",
            active_course="MATH 407 • Probability Theory",
        )
        results = recommend_resources("supervised and unsupervised machine learning", profile, limit=3)
        self.assertTrue(results)
        self.assertEqual(results[0].resource.title, "CS229 Machine Learning")
        self.assertFalse(any("active-course" in reason for item in results for reason in item.reasons))


if __name__ == "__main__":
    unittest.main()

