import unittest

from grillme.agents.conductor import conduct_turn
from grillme.agents.evaluator import evaluate_answer
from grillme.agents.question_planner import plan_next_question
from grillme.models.resume import ResumeProfile
from grillme.models.jd import JDProfile
from grillme.models.strategy import InterviewStrategy, QuestionHook
from grillme.models.interview import ConductorResponse, Evaluation, QuestionPlan


class Phase3InterviewLoopTest(unittest.TestCase):
    """Test Phase 3: Interview loop with 3 parallel agents (Evaluator ∥ Planner → Conductor)."""

    def setUp(self):
        """Set up test data."""
        self.strategy = InterviewStrategy(
            matching_skills=["Python", "ML"],
            missing_skills=["LangChain"],
            experience_gaps=[],
            transferable_strengths=["Architecture thinking"],
            question_hooks=[
                QuestionHook(
                    source="resume_project",
                    detail="Claims 94% accuracy — probe methodology",
                    question_type="ml_concepts",
                    priority="high"
                )
            ],
            opening_question_suggestion="Tell me about your most impactful ML project.",
            high_priority_topics=["Model Design", "Production ML"],
            company_focus_areas=["Scalability", "Real-time systems"],
            candidate_strengths_to_test=["ML Architecture", "System Design"],
        )

    def test_conductor_delivers_question_and_feedback(self):
        """Test that conductor agent delivers questions and feedback naturally."""
        response = conduct_turn(
            company="AI Corp",
            role="Senior ML Engineer",
            experience_tier="senior",
            difficulty="hard",
            strategy=self.strategy,
            question_number=1,
            questions_asked_count=0,
            type_coverage={},
            weak_areas=[],
            next_question="Tell me about model deployment strategies.",
            evaluation_summary=None,
            tone="balanced",
        )
        
        self.assertIsInstance(response, ConductorResponse)
        self.assertIsNotNone(response.response_text)
        self.assertGreater(len(response.response_text), 0)

    def test_evaluator_scores_answer(self):
        """Test that evaluator agent scores user answers."""
        user_answer = """
        I built a model using LightGBM with careful feature engineering.
        We achieved 0.892 AUC on the validation set through cross-validation.
        For deployment, we containerized with Docker and used Kubernetes for scaling.
        """
        
        evaluation = evaluate_answer(
            question_type="ml_concepts",
            question_text="Tell me about your most impactful ML project.",
            difficulty="hard",
            experience_tier="senior",
            company="AI Corp",
            role="Senior ML Engineer",
            user_answer=user_answer,
        )
        
        self.assertIsInstance(evaluation, Evaluation)
        self.assertGreaterEqual(evaluation.score, 0)
        self.assertLessEqual(evaluation.score, 10)
        self.assertGreater(len(evaluation.strengths), 0)
        self.assertGreater(len(evaluation.weaknesses), 0)

    def test_evaluator_handles_code_submission(self):
        """Test that evaluator handles code submissions for coding questions."""
        user_answer = "Implemented a sliding window approach with memoization."
        user_code = """
def maxWindow(arr, k):
    if not arr or k > len(arr):
        return 0
    
    max_sum = sum(arr[:k])
    current_sum = max_sum
    
    for i in range(1, len(arr) - k + 1):
        current_sum = current_sum - arr[i-1] + arr[i+k-1]
        max_sum = max(max_sum, current_sum)
    
    return max_sum
"""
        
        evaluation = evaluate_answer(
            question_type="coding",
            question_text="Implement a function to find the maximum sum of a subarray of size k.",
            difficulty="medium",
            experience_tier="junior",
            company="TechCorp",
            role="Software Engineer",
            user_answer=user_answer,
            user_code=user_code,
        )
        
        self.assertIsInstance(evaluation, Evaluation)
        self.assertIsNotNone(evaluation.code_feedback)
        self.assertIn(evaluation.code_feedback.correctness, ["correct", "partially correct", "incorrect"])

    def test_question_planner_selects_strategically(self):
        """Test that question planner selects next question based on coverage and weak areas."""
        type_coverage = {"behavioral": 2, "ml_concepts": 1}
        type_scores = {
            "behavioral": [7, 8],
            "ml_concepts": [5],
        }
        weak_areas = ["Model Evaluation", "Hyperparameter Tuning"]
        
        question_plan = plan_next_question(
            questions_asked_count=3,
            type_coverage=type_coverage,
            type_scores=type_scores,
            weak_areas=weak_areas,
            topics_asked=["Career Path", "Architecture Thinking"],
            follow_up_depth=0,
            difficulty="hard",
            experience_tier="senior",
            strategy=self.strategy,
            company="AI Corp",
        )
        
        self.assertIsInstance(question_plan, QuestionPlan)
        self.assertIsNotNone(question_plan.question_type)
        self.assertIsNotNone(question_plan.topic)
        self.assertIsNotNone(question_plan.reasoning)

    def test_interview_loop_flow(self):
        """Test a simulated interview loop turn: user answer → evaluate ∥ plan → conduct."""
        user_answer = "I used ensemble methods combining XGBoost and neural networks."
        
        # Step 1: Evaluate the answer
        evaluation = evaluate_answer(
            question_type="ml_concepts",
            question_text="How do you approach model selection?",
            difficulty="hard",
            experience_tier="senior",
            company="AI Corp",
            role="ML Engineer",
            user_answer=user_answer,
        )
        
        # Step 2: Plan next question (independent of evaluation)
        type_scores = {"ml_concepts": [evaluation.score]}
        question_plan = plan_next_question(
            questions_asked_count=1,
            type_coverage={"ml_concepts": 1},
            type_scores=type_scores,
            weak_areas=evaluation.weak_signals,
            topics_asked=["Model Selection"],
            follow_up_depth=0,
            difficulty="hard",
            experience_tier="senior",
            strategy=self.strategy,
            company="AI Corp",
        )
        
        # Step 3: Conductor delivers feedback + next question
        eval_summary = f"Score: {evaluation.score}/10. Strengths: {', '.join(evaluation.strengths)}"
        response = conduct_turn(
            company="AI Corp",
            role="ML Engineer",
            experience_tier="senior",
            difficulty="hard",
            strategy=self.strategy,
            question_number=2,
            questions_asked_count=1,
            type_coverage={"ml_concepts": 1},
            weak_areas=evaluation.weak_signals,
            next_question=f"Let's explore {question_plan.topic}",
            evaluation_summary=eval_summary,
        )
        
        # Verify the loop completed
        self.assertIsInstance(evaluation, Evaluation)
        self.assertIsInstance(question_plan, QuestionPlan)
        self.assertIsInstance(response, ConductorResponse)
        self.assertGreater(response.response_text.__len__(), 0)


if __name__ == "__main__":
    unittest.main()