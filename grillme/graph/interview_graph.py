"""LangGraph for GrillMe interview workflow — all phases."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from grillme.models.state import GrillMeState
from grillme.graph.nodes import (
    # Phase 2: Setup
    analyze_resume_node,
    analyze_jd_node,
    build_strategy_node,
    # Phase 3: Interview Loop
    generate_first_question_node,
    wait_for_input_node,
    evaluate_answer_node,
    plan_next_question_node,
    conduct_turn_node,
    check_continue_node,
)


def create_interview_graph():
    """
    Create the complete GrillMe interview graph with all phases.
    
    Phase 2 Setup (parallel fan-out/fan-in):
    ========================================
                        ┌────────────────┐
                  ┌────▶│ Analyze Resume │────┐
                  │     └────────────────┘     │
    START ────────┤                            ├──▶ Build Strategy ──┐
                  │     ┌────────────────┐     │                     │
                  └────▶│ Analyze JD     │────┘                     │
                        └────────────────┘                           │
                                                                     ▼
    Phase 3 Interview Loop (parallel evaluator ∥ planner):        First Q
                                                                     │
                                                                     ▼
                                                           ┌──────────────────┐
                                                           │ Wait for Input   │
                                                           │ (interrupt point)│
                                                           └────────┬─────────┘
                                                                    │ (user answers)
                                                                    ▼
                                                           ┌──────────────────────────┐
                                                  ┌──────▶│ Evaluate ∥ Plan Next    │
                                                  │       │ (parallel, no dependency)│
                                                  │       └────────────┬────────────┘
                                                  │                    │
                                    ┌─────────────┴────────────────────┘
                                    │
                                    ▼
                                ┌──────────────┐
                                │  Conduct     │
                                │  (feedback + │
                                │   next Q)    │
                                └──────┬───────┘
                                       │
                                       ▼
                                 ┌─────────────┐
                                 │Check Continue
                                 │(end or loop)│
                                 └──────┬──────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │ continue     │ end          │
                         ▼              ▼
                   ┌──────────────┐  ┌────────────────┐
                   │Wait for Input│  │Generate Report │
                   └──────┬───────┘  └────────┬───────┘
                          │                    │
                    (loop back)                ▼
                          │                   END
                          └────────────────────┘
    """
    builder = StateGraph(GrillMeState)
    
    # ========================
    # Phase 2: Setup Agents
    # ========================
    builder.add_node("analyze_resume", analyze_resume_node)
    builder.add_node("analyze_jd", analyze_jd_node)
    builder.add_node("build_strategy", build_strategy_node)
    
    # Fan-out: START triggers both resume and jd in parallel
    builder.add_edge(START, "analyze_resume")
    builder.add_edge(START, "analyze_jd")
    
    # Fan-in: both must complete before strategy runs
    builder.add_edge("analyze_resume", "build_strategy")
    builder.add_edge("analyze_jd", "build_strategy")
    
    # ========================
    # Phase 3: Interview Loop
    # ========================
    builder.add_node("generate_first_question", generate_first_question_node)
    builder.add_node("wait_for_input", wait_for_input_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("plan_next_question", plan_next_question_node)
    builder.add_node("conduct_turn", conduct_turn_node)
    builder.add_node("check_continue", check_continue_node)
    
    # Setup → First Question
    builder.add_edge("build_strategy", "generate_first_question")
    builder.add_edge("generate_first_question", "wait_for_input")
    
    # Interview loop: after user input, parallel fan-out (evaluator ∥ planner)
    builder.add_edge("wait_for_input", "evaluate_answer")
    builder.add_edge("wait_for_input", "plan_next_question")
    
    # Fan-in: both must complete before conductor
    builder.add_edge("evaluate_answer", "conduct_turn")
    builder.add_edge("plan_next_question", "conduct_turn")
    
    # After conductor, check if we continue or end
    builder.add_edge("conduct_turn", "check_continue")
    
    # Conditional routing
    builder.add_conditional_edges(
        "check_continue",
        lambda state: "end" if state.get("should_end", False) or state.get("question_count", 0) >= 15 else "continue",
        {"continue": "wait_for_input", "end": END}
    )
    
    # Compile with memory saver for checkpointing
    graph = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["wait_for_input"]
    )
    return graph


# Singleton graph instance
interview_graph = create_interview_graph()
