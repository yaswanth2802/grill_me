import unittest

from grillme.config import get_main_llm
from grillme.models.resume import ResumeProfile
from grillme.models.jd import JDProfile
from grillme.models.strategy import InterviewStrategy


class Phase1ScaffoldingTest(unittest.TestCase):
    def test_config_module_exposes_llm_factory(self):
        # Phase 1: just verify the config module can be imported and called
        # Actual LLM instantiation requires API keys (Phase 2+)
        try:
            llm = get_main_llm()
            self.assertTrue(llm is None or hasattr(llm, "invoke"))
        except Exception as e:
            # API key missing is expected in Phase 1 scaffold
            self.assertIn("API key", str(e))

    def test_models_can_be_instantiated(self):
        resume = ResumeProfile(
            name="Ada Lovelace",
            skills=["Python", "ML"],
            experience_entries=[],
            projects=[],
            education=[],
            certifications=[],
            total_experience_years=3.0,
            domain_expertise=["software"],
        )
        jd = JDProfile(
            company="Acme",
            role="ML Engineer",
            required_skills=["Python", "ML"],
            preferred_skills=["LangChain"],
            responsibilities=["build models"],
            experience_required="2+ years",
            domain="AI",
        )
        strategy = InterviewStrategy(
            matching_skills=["Python"],
            missing_skills=["LangChain"],
            experience_gaps=[],
            transferable_strengths=["ML"],
            question_hooks=[],
            opening_question_suggestion="Tell me about a project you shipped.",
            high_priority_topics=["model design"],
            company_focus_areas=["product"],
            candidate_strengths_to_test=["ML"],
        )

        self.assertEqual(resume.name, "Ada Lovelace")
        self.assertEqual(jd.role, "ML Engineer")
        self.assertEqual(strategy.opening_question_suggestion, "Tell me about a project you shipped.")


if __name__ == "__main__":
    unittest.main()
