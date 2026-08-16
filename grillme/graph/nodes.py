"""Node functions for the GrillMe LangGraph."""

from grillme.agents.resume_analyzer import analyze_resume
from grillme.agents.jd_analyzer import analyze_jd
from grillme.agents.strategy_agent import build_interview_strategy
from grillme.agents.conductor import conduct_turn
from grillme.agents.evaluator import evaluate_answer
from grillme.agents.question_planner import plan_next_question
from grillme.models.state import GrillMeState


# ============================================================================
# SETUP PHASE NODES (Phase 2)
# ============================================================================

def analyze_resume_node(state: GrillMeState) -> dict:
    """Resume Analyzer node — extracts structured profile from resume text."""
    print("[node:analyze_resume_node] ENTRY")
    resume_text = state.get("resume_text", "")
    if not resume_text:
        raise ValueError("resume_text is required in state")
    
    profile = analyze_resume(resume_text)
    try:
        out = profile.model_dump_json()
    except Exception:
        out = str(profile)
    print(f"[node:analyze_resume_node] EXIT resume_profile={out}")
    return {"resume_profile": profile}


def analyze_jd_node(state: GrillMeState) -> dict:
    """JD Analyzer node — extracts structured profile from job description."""
    print("[node:analyze_jd_node] ENTRY")
    jd_text = state.get("jd_text", "")
    if not jd_text:
        raise ValueError("jd_text is required in state")
    
    profile = analyze_jd(jd_text)
    try:
        out = profile.model_dump_json()
    except Exception:
        out = str(profile)
    print(f"[node:analyze_jd_node] EXIT jd_profile={out}")
    return {"jd_profile": profile}


def build_strategy_node(state: GrillMeState) -> dict:
    """Strategy Agent node — creates interview strategy from resume + JD profiles."""
    print("[node:build_strategy_node] ENTRY")
    resume_profile = state.get("resume_profile")
    jd_profile = state.get("jd_profile")
    
    if resume_profile is None:
        raise ValueError("resume_profile is required in state")
    if jd_profile is None:
        raise ValueError("jd_profile is required in state")
    
    strategy = build_interview_strategy(resume_profile, jd_profile)
    try:
        out = strategy.model_dump_json()
    except Exception:
        out = str(strategy)
    print(f"[node:build_strategy_node] EXIT interview_strategy={out}")
    return {"interview_strategy": strategy}


# ============================================================================
# INTERVIEW LOOP NODES (Phase 3)
# ============================================================================

def generate_first_question_node(state: GrillMeState) -> dict:
    """Generate the first interview question using the strategy."""
    print("[node:generate_first_question_node] ENTRY")
    strategy = state.get("interview_strategy")
    if strategy is None:
        raise ValueError("interview_strategy is required in state")
    out = {"current_question_type": "behavioral", "question_count": 1}
    print(f"[node:generate_first_question_node] EXIT {out}")
    return out


from langgraph.types import interrupt

def wait_for_input_node(state: GrillMeState):
    """Node that pauses execution and waits for user input from Streamlit."""
    # The interrupt call pauses execution and waits for the Command(resume=...) payload
    user_input = interrupt("Waiting for user answer...")
    
    # CRITICAL: You must return the received user input so LangGraph updates the state channels!
    return {
        "user_answer": user_input.get("user_answer"),
        "user_code": user_input.get("user_code"),
        "should_end": user_input.get("should_end", False),
        "question_count": user_input.get("question_count", state.get("question_count", 0)),
    }


def evaluate_answer_node(state: GrillMeState) -> dict:
    """Evaluator node — scores the user's answer (runs in parallel with planner)."""
    print("[node:evaluate_answer_node] ENTRY")
    question_type = state.get("current_question_type", "behavioral")
    user_answer = state.get("user_answer", "")
    user_code = state.get("user_code")
    experience_tier = state.get("experience_tier", "junior")
    difficulty = state.get("difficulty", "medium")
    company = state.get("company", "Unknown")
    
    conversation = state.get("conversation_history", [])
    question_text = "Tell me about your experience."
    if conversation:
        for msg in reversed(conversation):
            if msg.get("role") == "interviewer":
                question_text = msg.get("content", "")
                break
    
    jd = state.get("jd_profile")
    role = jd.role if jd else "Engineer"
    
    evaluation = evaluate_answer(
        question_type=question_type,
        question_text=question_text,
        difficulty=difficulty,
        experience_tier=experience_tier,
        company=company,
        role=role,
        user_answer=user_answer,
        user_code=user_code,
    )
    
    try:
        out = evaluation.model_dump_json()
    except Exception:
        out = str(evaluation)
    print(f"[node:evaluate_answer_node] EXIT current_evaluation={out}")
    return {"current_evaluation": evaluation}


def plan_next_question_node(state: GrillMeState) -> dict:
    """Question Planner node — selects next question (runs in parallel with evaluator)."""
    print("[node:plan_next_question_node] ENTRY")
    question_records = state.get("question_records", [])
    type_coverage = state.get("type_coverage", {})
    type_scores = state.get("type_scores", {})
    weak_areas = state.get("weak_areas", [])
    topics_asked = state.get("topics_asked", [])
    strategy = state.get("interview_strategy")
    company = state.get("company", "Unknown")
    
    # ── RETRIEVE THE 3 MISSING PARAMETERS FROM STATE ──
    follow_up_depth = state.get("follow_up_depth", 0)
    difficulty = state.get("difficulty", "medium")
    experience_tier = state.get("experience_tier", "junior")
    
    if strategy is None:
        raise ValueError("interview_strategy is required in state")
    
    questions_asked = len(question_records)
    
    question_plan = plan_next_question(
        questions_asked_count=questions_asked,
        type_coverage=type_coverage,
        type_scores=type_scores,
        weak_areas=weak_areas,
        topics_asked=topics_asked,
        follow_up_depth=follow_up_depth,       # Passed here
        difficulty=difficulty,                 # Passed here
        experience_tier=experience_tier,       # Passed here
        strategy=strategy,
        company=company,
    )
    
    try:
        out = question_plan.model_dump_json()
    except Exception:
        out = str(question_plan)
    print(f"[node:plan_next_question_node] EXIT next_question_plan={out}")
    return {"next_question_plan": question_plan}


def conduct_turn_node(state: GrillMeState) -> dict:
    """Conductor node — delivers feedback and next question (runs after evaluator + planner)."""
    print("[node:conduct_turn_node] ENTRY")
    evaluation = state.get("current_evaluation")
    question_plan = state.get("next_question_plan")
    strategy = state.get("interview_strategy")
    company = state.get("company", "Unknown")
    difficulty = state.get("difficulty", "medium")
    experience_tier = state.get("experience_tier", "junior")
    question_records = state.get("question_records", [])
    type_coverage = state.get("type_coverage", {})
    weak_areas = state.get("weak_areas", [])
    feedback_mode = state.get("feedback_mode", "after_each")
    
    jd = state.get("jd_profile")
    role = jd.role if jd else "Engineer"
    
    eval_summary = ""
    if evaluation and feedback_mode == "after_each":
        eval_summary = f"Score: {evaluation.score}/10. Strengths: {', '.join(evaluation.strengths)}. Areas to improve: {', '.join(evaluation.weaknesses)}"
    
    next_question_text = f"{question_plan.topic}" if question_plan else None
    
    response = conduct_turn(
        company=company,
        role=role,
        experience_tier=experience_tier,
        difficulty=difficulty,
        strategy=strategy,
        question_number=len(question_records) + 1,
        questions_asked_count=len(question_records),
        type_coverage=type_coverage,
        weak_areas=weak_areas,
        next_question=next_question_text,
        evaluation_summary=eval_summary,
    )
    
    try:
        out = response.model_dump_json()
    except Exception:
        out = str(response)
    print(f"[node:conduct_turn_node] EXIT agent_response={out}")
    return {"agent_response": response}


def check_continue_node(state: GrillMeState) -> str:
    """Conditional node — determines if interview continues or ends."""
    should_end = state.get("should_end", False)
    question_count = state.get("question_count", 0)
    max_questions = 15
    
    # if should_end or question_count >= max_questions:
    #     return "end"
    
    # return "continue"
    return {}
