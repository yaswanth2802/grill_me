import unittest

from grillme.agents.resume_analyzer import analyze_resume
from grillme.agents.jd_analyzer import analyze_jd
from grillme.agents.strategy_agent import build_interview_strategy
from grillme.graph.interview_graph import interview_graph
from grillme.models.resume import ResumeProfile
from grillme.models.jd import JDProfile
from grillme.models.strategy import InterviewStrategy


class Phase2SetupAgentsTest(unittest.TestCase):
    """Test Phase 2: Parallel Resume + JD → Strategy setup agents."""

    def setUp(self):
        """Set up test data."""
        self.resume_text = """
        Name: Alice Chen
        
        Skills: Python, Golang, Machine Learning, TensorFlow, PostgreSQL, Docker, Kubernetes
        
        Experience:
        - Senior ML Engineer at TechCorp (2022-2024, 2 years)
          • Designed and deployed ML pipeline reducing inference latency by 60%
          • Led team of 3 engineers on recommender system
          • Tech: Python, TensorFlow, Kubernetes
        
        - ML Engineer at DataInc (2020-2022, 2 years)
          • Built and trained models achieving 94% accuracy on classification task
          • Tech: Python, scikit-learn, PostgreSQL
        
        Education: MS Computer Science from Stanford University
        Certifications: AWS ML Specialty, TensorFlow Certified Developer
        """
        
        self.jd_text = """
        Job Title: Senior Machine Learning Engineer
        Company: AI Innovations Inc
        
        Required Skills:
        - Python (expert level)
        - Machine Learning (5+ years)
        - Deep Learning frameworks (TensorFlow, PyTorch)
        - System design
        
        Preferred Skills:
        - Kubernetes, Docker
        - Production ML systems
        - Leadership experience
        
        Responsibilities:
        - Design and implement ML systems at scale
        - Mentor junior engineers
        - Collaborate with product teams
        
        Experience Required: 5+ years in ML
        Domain: AI/ML
        """

    def test_resume_analyzer_returns_structured_profile(self):
        """Test that resume analyzer extracts structured profile."""
        profile = analyze_resume(self.resume_text)
        
        self.assertIsInstance(profile, ResumeProfile)
        self.assertIsNotNone(profile.name)
        self.assertGreater(len(profile.skills), 0)
        self.assertGreater(len(profile.experience_entries), 0)
        self.assertGreater(profile.total_experience_years, 0)

    def test_jd_analyzer_returns_structured_profile(self):
        """Test that JD analyzer extracts structured profile."""
        profile = analyze_jd(self.jd_text)
        
        self.assertIsInstance(profile, JDProfile)
        self.assertEqual(profile.role, "Senior Machine Learning Engineer")
        self.assertEqual(profile.company, "AI Innovations Inc")
        self.assertGreater(len(profile.required_skills), 0)
        self.assertGreater(len(profile.preferred_skills), 0)

    def test_strategy_agent_generates_interview_strategy(self):
        """Test that strategy agent generates gap analysis and interview plan."""
        resume = analyze_resume(self.resume_text)
        jd = analyze_jd(self.jd_text)
        strategy = build_interview_strategy(resume, jd)
        
        self.assertIsInstance(strategy, InterviewStrategy)
        self.assertGreater(len(strategy.matching_skills), 0)
        self.assertGreater(len(strategy.high_priority_topics), 0)
        self.assertIsNotNone(strategy.opening_question_suggestion)

    def test_graph_parallel_fan_out_fan_in(self):
        """Test that the graph runs with parallel resume/jd analysis."""
        state_input = {
            "resume_text": self.resume_text,
            "jd_text": self.jd_text,
        }
        
        # Run the graph — should execute analyze_resume and analyze_jd in parallel,
        # then build_strategy after both complete
        config = {"configurable": {"thread_id": "test-phase-2"}}
        output = interview_graph.invoke(state_input, config=config)
        
        # Verify all three agents ran
        self.assertIn("resume_profile", output)
        self.assertIn("jd_profile", output)
        self.assertIn("interview_strategy", output)
        
        self.assertIsInstance(output["resume_profile"], ResumeProfile)
        self.assertIsInstance(output["jd_profile"], JDProfile)
        self.assertIsInstance(output["interview_strategy"], InterviewStrategy)


if __name__ == "__main__":
    unittest.main()
