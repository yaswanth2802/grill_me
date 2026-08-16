# GrillMe — Implementation Plan (LangChain + LangGraph Multi-Agent)

## Tech Stack Summary

| Layer | Choice | Reason |
|---|---|---|
| **Frontend** | Streamlit | Fast to build, Python-native, good enough for MVP |
| **Orchestration** | LangGraph | Multi-agent state machine with parallel fan-out, conditional edges, interrupts |
| **LLM Abstraction** | LangChain chat models | `ChatGoogleGenerativeAI`, `ChatOpenAI`, `ChatAnthropic` — swap via `.env` |
| **Structured Output** | Pydantic + `with_structured_output()` | Type-safe LLM responses, auto-retry on parse failure |
| **LLM Default** | Google Gemini (configurable) | User preference; model name + API key via `.env` |
| **STT** | `faster-whisper` (local, open-source) | Free, runs locally, no API key needed |
| **TTS** | `edge-tts` (Microsoft Edge TTS) | Free, high-quality voices, async, no API key |
| **Resume Parsing** | `pdfplumber` | Reliable PDF text extraction with layout awareness |
| **State Management** | LangGraph state + Streamlit `session_state` | Graph state is the source of truth; Streamlit bridges to UI |
| **Deployment** | Local only | MVP — deploy later |

---

## Architecture: Multi-Agent LangGraph

### Agent Inventory

Six specialized agents. The key insight: agents that don't depend on each other run **in parallel**.

| Agent | What It Does | Input | Output (Pydantic) |
|---|---|---|---|
| **Resume Analyzer** | Extracts structured profile from resume PDF | Raw resume text | `ResumeProfile` |
| **JD Analyzer** | Extracts structured profile from job description | JD text | `JDProfile` |
| **Strategy Agent** | Compares resume vs JD → interview strategy + resume improvement advice | `ResumeProfile` + `JDProfile` | `InterviewStrategy` |
| **Conductor** | Speaks to user — delivers questions, responds naturally, gives feedback | Strategy + conversation + next question plan | `ConductorResponse` |
| **Evaluator** | Scores the user's answer (can run on cheaper/faster model) | Question + answer + rubric | `Evaluation` |
| **Question Planner** | Picks the next question type + topic based on coverage, scores, weak areas | All `QuestionRecord`s + strategy | `QuestionPlan` |
| **Report Generator** | Synthesizes full session into a scored report | All `QuestionRecord`s + strategy | `GrindReport` |

### Parallelism Map

```
SETUP PHASE — fan-out / fan-in
═══════════════════════════════════════════════════════════

                    ┌───────────────────┐
              ┌────▶│  Resume Analyzer   │────┐
              │     └───────────────────┘     │
  START ──────┤                               ├──▶ Strategy Agent ──▶ First Question
              │     ┌───────────────────┐     │
              └────▶│  JD Analyzer       │────┘
                    └───────────────────┘

  Resume + JD run IN PARALLEL (no dependency)
  Strategy Agent waits for BOTH (fan-in)


INTERVIEW LOOP — 3 agents per turn, 2 run in parallel
═══════════════════════════════════════════════════════════

  User submits answer
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  Fan-out: Evaluator ∥ Question Planner   │  ◀── run in PARALLEL
  │                                          │      (both read user's answer,
  │  ┌─────────────┐   ┌──────────────────┐  │       neither depends on the other)
  │  │  Evaluator   │   │ Question Planner │  │
  │  │  scores the  │   │ picks next type  │  │
  │  │  answer      │   │ + topic + hook   │  │
  │  └──────┬──────┘   └───────┬──────────┘  │
  │         │                   │              │
  │         └─────────┬─────────┘              │
  │                   │ fan-in                 │
  └───────────────────┼────────────────────────┘
                      ▼
               ┌──────────────┐
               │  Conductor    │  ◀── takes evaluation + question plan
               │  speaks to    │      delivers feedback (if enabled) + next question
               │  the user     │
               └──────┬───────┘
                      │
                      ▼
               [WAIT FOR USER INPUT]  ◀── graph interrupts here
                      │
                      ├── user answers ──▶ loop back to fan-out
                      └── "End Grind" ──▶ Report Generator ──▶ END


POST-GRIND — Strategy Agent reuse
═══════════════════════════════════════════════════════════

  User clicks "Resume Advice"
         │
         ▼
  Strategy Agent (resume_advice mode)
         │
         ▼
  Returns: ResumeAdvice
    - skills to add/highlight
    - projects to emphasize
    - gaps to address
    - phrasing suggestions
    - tailored bullet points for target JD
```

### Why This Parallelism Matters

| Sequential (old) | Parallel (new) | Speedup |
|---|---|---|
| Resume → JD → Gap (3 LLM calls serial) | Resume ∥ JD → Strategy (2 serial) | ~33% faster setup |
| Evaluate → Plan → Conduct (3 serial per turn) | (Evaluate ∥ Plan) → Conduct (2 serial per turn) | ~33% faster per turn |
| Over 15 questions, that's 15 saved LLM calls of latency | | Significant UX improvement |

---

### LangGraph State

```python
class GrillMeState(TypedDict):
    # Setup inputs
    resume_text: str
    jd_text: str
    experience_tier: str              # "student" | "junior" | "senior"
    difficulty: str                   # "easy" | "medium" | "hard"
    question_types: list[str]         # ["behavioral", "system_design", ...]
    feedback_mode: str                # "after_each" | "final_only"
    interaction_mode: str             # "voice" | "chat"
    company: str

    # Analysis results (populated by setup agents)
    resume_profile: ResumeProfile | None
    jd_profile: JDProfile | None
    interview_strategy: InterviewStrategy | None

    # Interview state
    conversation_history: list[dict]
    question_records: list[QuestionRecord]
    current_question_type: str | None
    type_coverage: dict[str, int]
    type_scores: dict[str, list[float]]
    weak_areas: list[str]
    topics_asked: list[str]
    follow_up_depth: int
    session_start_time: str
    question_count: int
    should_end: bool

    # Per-turn parallel outputs (written by Evaluator ∥ Planner, read by Conductor)
    user_answer: str
    user_code: str | None
    current_evaluation: Evaluation | None       # from Evaluator
    next_question_plan: QuestionPlan | None      # from Question Planner
    agent_response: str                          # from Conductor

    # Report
    report: GrindReport | None

    # Resume advice (optional, triggered by user)
    resume_advice: ResumeAdvice | None
```

### Graph Definition (Pseudocode)

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(GrillMeState)

# ── Setup phase (parallel) ──
builder.add_node("analyze_resume", analyze_resume_node)
builder.add_node("analyze_jd", analyze_jd_node)
builder.add_node("build_strategy", build_strategy_node)
builder.add_node("generate_first_question", generate_first_question_node)

# Fan-out: START → resume ∥ jd
builder.add_edge(START, "analyze_resume")
builder.add_edge(START, "analyze_jd")
# Fan-in: both → strategy
builder.add_edge("analyze_resume", "build_strategy")
builder.add_edge("analyze_jd", "build_strategy")
builder.add_edge("build_strategy", "generate_first_question")

# ── Interview loop ──
builder.add_node("wait_for_input", wait_for_input_node)      # interrupt point
builder.add_node("evaluate_answer", evaluate_answer_node)      # Evaluator agent
builder.add_node("plan_next_question", plan_next_question_node) # Question Planner agent
builder.add_node("conduct_turn", conduct_turn_node)            # Conductor agent
builder.add_node("check_continue", check_continue_node)

builder.add_edge("generate_first_question", "wait_for_input")

# After user input: fan-out evaluator ∥ planner
builder.add_edge("wait_for_input", "evaluate_answer")
builder.add_edge("wait_for_input", "plan_next_question")
# Fan-in: both → conductor
builder.add_edge("evaluate_answer", "conduct_turn")
builder.add_edge("plan_next_question", "conduct_turn")

builder.add_edge("conduct_turn", "check_continue")

builder.add_conditional_edges(
    "check_continue",
    should_continue,
    {"continue": "wait_for_input", "end": "generate_report"}
)

# ── Report ──
builder.add_node("generate_report", generate_report_node)
builder.add_edge("generate_report", END)

# ── Compile with interrupt ──
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["wait_for_input"]
)
```

---

## Project Structure

```
grillme/
├── .env                              # API keys + model config
├── .env.example                      # Template
├── requirements.txt
├── app.py                            # Streamlit entry point
├── config.py                         # Loads .env, creates LLM instances
│
├── models/                           # Pydantic schemas (shared across agents)
│   ├── __init__.py
│   ├── resume.py                     # ResumeProfile, ExperienceEntry, Project
│   ├── jd.py                         # JDProfile
│   ├── strategy.py                   # InterviewStrategy, ResumeAdvice
│   ├── interview.py                  # QuestionRecord, Evaluation, QuestionPlan, ConductorResponse
│   ├── report.py                     # GrindReport, QuestionDetail, WeaknessPattern
│   └── state.py                      # GrillMeState (TypedDict for LangGraph)
│
├── agents/                           # Individual agent implementations
│   ├── __init__.py
│   ├── resume_analyzer.py            # Resume extraction
│   ├── jd_analyzer.py                # JD extraction
│   ├── strategy_agent.py             # Gap analysis + interview strategy + resume advice
│   ├── conductor.py                  # Interviewer voice — delivers questions + feedback
│   ├── evaluator.py                  # Scores answers (can use cheaper model)
│   ├── question_planner.py           # Adaptive next-question selection
│   └── report_generator.py           # Report synthesis
│
├── graph/                            # LangGraph wiring
│   ├── __init__.py
│   ├── interview_graph.py            # Full graph: nodes, edges, fan-out/fan-in
│   └── nodes.py                      # Node functions bridging agents ↔ state
│
├── prompts/                          # System prompts per agent
│   ├── __init__.py
│   ├── resume_analyzer.py
│   ├── jd_analyzer.py
│   ├── strategy_agent.py
│   ├── conductor.py                  # Dynamic prompt (tier, company, tone)
│   ├── evaluator.py
│   ├── question_planner.py
│   └── report_generator.py
│
├── voice/
│   ├── __init__.py
│   ├── stt.py                        # faster-whisper wrapper
│   └── tts.py                        # edge-tts wrapper
│
├── ui/
│   ├── __init__.py
│   ├── setup_page.py                 # Resume upload, JD, settings
│   ├── grind_page.py                 # Interview UI (chat + voice + code)
│   ├── report_page.py                # Report display
│   └── resume_advice_page.py         # Resume improvement suggestions
│
└── data/
    └── company_profiles.json         # Company-specific topic adjustments
```

---

## `.env` Configuration

```env
# LLM Config (required)
LLM_PROVIDER=gemini                    # gemini | openai | anthropic
LLM_MODEL=gemini-2.0-flash            # model for Conductor, Strategy, Planner, Report
LLM_API_KEY=your-gemini-api-key

# Optional: cheaper/faster model for Evaluator agent
EVALUATOR_MODEL=gemini-2.0-flash-lite  # can be a smaller/cheaper model
EVALUATOR_API_KEY=                     # leave blank to use LLM_API_KEY

# Optional: secondary model for experimentation
LLM_MODEL_ALT=gemini-1.5-pro
LLM_API_KEY_ALT=another-key

# Voice (optional)
WHISPER_MODEL_SIZE=base                # tiny | base | small | medium | large-v3
TTS_VOICE=en-US-AriaNeural            # edge-tts voice name
```

---

## Pydantic Models (Complete Definitions)

### `models/resume.py`

```python
class ExperienceEntry(BaseModel):
    role: str
    company: str
    duration: str
    highlights: list[str]
    technologies: list[str]

class Project(BaseModel):
    name: str
    description: str
    metrics: list[str]
    technologies: list[str]

class ResumeProfile(BaseModel):
    name: str
    skills: list[str]
    experience_entries: list[ExperienceEntry]
    projects: list[Project]
    education: list[str]
    certifications: list[str]
    total_experience_years: float
    domain_expertise: list[str]
```

### `models/jd.py`

```python
class JDProfile(BaseModel):
    company: str
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    experience_required: str
    domain: str
```

### `models/strategy.py` (NEW — expanded from old GapAnalysis)

```python
class InterviewStrategy(BaseModel):
    """Generated by Strategy Agent — used by all interview agents."""
    # Gap analysis
    matching_skills: list[str]
    missing_skills: list[str]
    experience_gaps: list[str]
    transferable_strengths: list[str]

    # Interview strategy (new — this is what makes the agent worthwhile)
    question_hooks: list[QuestionHook]     # specific things to probe, in priority order
    opening_question_suggestion: str       # tailored first behavioral question
    high_priority_topics: list[str]        # topics that MUST be covered
    company_focus_areas: list[str]         # adjusted for target company
    candidate_strengths_to_test: list[str] # verify claimed strengths aren't inflated

class QuestionHook(BaseModel):
    """A specific interview angle derived from resume/JD analysis."""
    source: str       # "resume_project" | "resume_metric" | "jd_gap" | "career_trajectory"
    detail: str       # "Claims 94% accuracy improvement — probe methodology"
    question_type: str # "behavioral" | "system_design" | "ml_concepts" | etc.
    priority: str     # "high" | "medium" | "low"

class ResumeAdvice(BaseModel):
    """Resume improvement suggestions for the target JD — reuses Strategy Agent."""
    skills_to_add: list[str]
    skills_to_highlight: list[str]
    projects_to_emphasize: list[str]
    gaps_to_address: list[str]
    bullet_point_suggestions: list[BulletSuggestion]
    overall_fit_score: float              # 0-10 how well resume matches JD
    summary: str

class BulletSuggestion(BaseModel):
    original: str | None                  # existing bullet (None if new)
    suggested: str                        # improved/new bullet
    reason: str                           # why this change
```

### `models/interview.py`

```python
class Evaluation(BaseModel):
    """Output of the Evaluator agent — runs in parallel with Question Planner."""
    score: float = Field(ge=0, le=10)
    strengths: list[str]
    weaknesses: list[str]
    missed_points: list[str]
    weak_signals: list[str]               # "lacks depth", "misses edge cases"
    code_feedback: CodeFeedback | None = None

class CodeFeedback(BaseModel):
    correctness: str                      # "correct" | "partially correct" | "incorrect"
    time_complexity: str
    space_complexity: str
    edge_cases_handled: bool
    quality_notes: list[str]
    optimal_approach_hint: str | None = None

class QuestionPlan(BaseModel):
    """Output of the Question Planner agent — runs in parallel with Evaluator."""
    question_type: str                    # "behavioral" | "system_design" | etc.
    is_follow_up: bool
    topic: str                            # specific topic label for anti-repetition
    directive: str                        # instruction to Conductor
    resume_hook: str | None = None        # specific resume item to reference
    difficulty_adjustment: str            # "same" | "harder" | "easier"
    reasoning: str                        # why this question was chosen (for debugging)

class ConductorResponse(BaseModel):
    """Output of the Conductor agent — speaks to the user."""
    interviewer_message: str              # what the user sees/hears
    feedback_given: str | None = None     # if feedback_mode = "after_each"
    is_coding_question: bool
    question_text_for_display: str | None = None  # for coding Qs, show text separately

class QuestionRecord(BaseModel):
    """Complete record of one Q&A exchange — stored in state."""
    question_text: str
    question_type: str
    question_topic: str
    user_answer: str
    user_code: str | None = None
    evaluation: Evaluation
    is_follow_up: bool
    turn_number: int
```

### `models/report.py`

```python
class QuestionDetail(BaseModel):
    question_text: str
    question_type: str
    user_answer_summary: str
    code_submitted: str | None = None
    score: float
    strengths: list[str]
    weaknesses: list[str]
    missed_points: list[str]
    is_follow_up: bool

class WeaknessPattern(BaseModel):
    pattern: str
    evidence: list[str]
    severity: str

class Recommendation(BaseModel):
    topic: str
    why: str
    suggested_resources: list[str]

class GrindReport(BaseModel):
    overall_score: float
    category_scores: dict[str, float]
    questions: list[QuestionDetail]
    weakness_patterns: list[WeaknessPattern]
    recommendations: list[Recommendation]
    session_duration_minutes: float
    total_questions: int
```

---

## Phase 1: Scaffolding, Config, and LLM Abstraction

**Goal:** Runnable Streamlit app with LangChain chat model calling Gemini, all Pydantic models defined, `with_structured_output()` verified.

### Tasks

1. **Initialize project structure** — all directories, `__init__.py` files, empty modules.

2. **`requirements.txt`**:
   ```
   streamlit>=1.35.0
   langgraph>=0.2.0
   langchain-core>=0.3.0
   langchain-google-genai>=2.0.0
   langchain-openai>=0.2.0
   langchain-anthropic>=0.2.0
   python-dotenv>=1.0.0
   pdfplumber>=0.11.0
   faster-whisper>=1.0.0
   edge-tts>=6.1.0
   pydantic>=2.0.0
   ```

3. **`config.py`** — LLM factory:
   ```python
   def create_llm(provider, model, api_key, temperature=0.7) -> BaseChatModel:
       if provider == "gemini":
           return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
       elif provider == "openai":
           return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
       elif provider == "anthropic":
           return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)

   def get_main_llm() -> BaseChatModel: ...
   def get_evaluator_llm() -> BaseChatModel: ...   # can be cheaper model
   ```

4. **`models/`** — all Pydantic schemas as defined above.

5. **`app.py`** — Streamlit entry, multi-page nav (Setup → Grind → Report → Resume Advice).

6. **Smoke test**: `llm.with_structured_output(ResumeProfile).invoke(...)` works with Gemini.

### Exit Criteria
- `streamlit run app.py` launches with placeholder pages.
- Structured output returns valid Pydantic object from Gemini.

### Phase 1 Status: ✅ SCAFFOLDING COMPLETE

**Completed:**
1. ✅ Project structure — all directories, `__init__.py` files initialized
2. ✅ `requirements.txt` — all dependencies listed (installed & verified)
3. ✅ `config.py` — LLM factory with `create_llm()`, `get_main_llm()`, `get_evaluator_llm()` functions
4. ✅ All Pydantic models:
   - `models/resume.py` — `ResumeProfile`, `ExperienceEntry`, `Project`
   - `models/jd.py` — `JDProfile`
   - `models/strategy.py` — `InterviewStrategy`, `QuestionHook`, `ResumeAdvice`, `BulletSuggestion`
   - `models/interview.py` — `Evaluation`, `CodeFeedback`, `QuestionRecord`, `QuestionPlan`, `ConductorResponse`
   - `models/report.py` — `GrindReport`, `QuestionDetail`, `WeaknessPattern`, `Recommendation`
   - `models/state.py` — `GrillMeState` TypedDict
5. ✅ `app.py` — Streamlit multi-page entry (Setup, Grind, Report, Resume Advice)
6. ✅ Test suite: `tests/test_phase1.py` — all model instantiation & config factory tests passing

**Next:** Phase 2 requires API key in `.env` (GOOGLE_API_KEY or similar). Set this before proceeding to implement agents.

---

## Phase 2: Setup Agents (Parallel Resume + JD → Strategy)

**Goal:** Three agents wired as fan-out/fan-in. Resume and JD analyze in parallel, Strategy Agent waits for both.

### Tasks

1. **`agents/resume_analyzer.py`**:
   - `llm.with_structured_output(ResumeProfile)`
   - Deep extraction: skills, projects with metrics, career trajectory, domain expertise
   - PDF → text via `pdfplumber` happens before this agent runs

2. **`agents/jd_analyzer.py`**:
   - `llm.with_structured_output(JDProfile)`
   - Company, role, skills, responsibilities, domain

3. **`agents/strategy_agent.py`** (expanded from old Gap Finder):
   - `llm.with_structured_output(InterviewStrategy)`
   - Takes serialized `ResumeProfile` + `JDProfile`
   - Produces:
     - Gap analysis (matching/missing skills, experience gaps)
     - **Interview strategy**: prioritized question hooks, opening question suggestion, high-priority topics, company focus areas, strengths to verify
   - **Second mode — resume advice**: same agent, different prompt, returns `ResumeAdvice`
   ```python
   class StrategyAgent:
       def analyze_for_interview(self, resume, jd) -> InterviewStrategy: ...
       def generate_resume_advice(self, resume, jd) -> ResumeAdvice: ...
   ```

4. **`prompts/`** — system prompts for all three agents.

5. **`graph/interview_graph.py`** — setup subgraph with **parallel fan-out**:
   ```python
   # Fan-out: START triggers both resume + jd in parallel
   builder.add_edge(START, "analyze_resume")
   builder.add_edge(START, "analyze_jd")
   # Fan-in: both must complete before strategy runs
   builder.add_edge("analyze_resume", "build_strategy")
   builder.add_edge("analyze_jd", "build_strategy")
   ```

6. **`graph/nodes.py`** — node functions:
   ```python
   def analyze_resume_node(state: GrillMeState) -> dict:
       profile = resume_analyzer.analyze(state["resume_text"])
       return {"resume_profile": profile}

   def analyze_jd_node(state: GrillMeState) -> dict:
       profile = jd_analyzer.analyze(state["jd_text"])
       return {"jd_profile": profile}

   def build_strategy_node(state: GrillMeState) -> dict:
       strategy = strategy_agent.analyze_for_interview(
           state["resume_profile"], state["jd_profile"]
       )
       return {"interview_strategy": strategy}
   ```

7. **`ui/setup_page.py`** — setup form:
   - File uploader (PDF), JD text area
   - Experience tier, difficulty, feedback mode, question types, interaction mode
   - "Start Grind" button → runs setup subgraph → shows progress:
     - "Analyzing resume..." + "Analyzing job description..." (shown simultaneously since they're parallel)
     - "Building interview strategy..."

### Exit Criteria
- Upload PDF + paste JD → Resume + JD agents run **in parallel** → Strategy Agent produces `InterviewStrategy`.
- Measurably faster than sequential (2 serial LLM calls instead of 3).
- `InterviewStrategy.question_hooks` contains resume-specific probing angles.

### Phase 2 Status: ✅ SETUP AGENTS COMPLETE

**Completed:**
1. ✅ `.env` copied from `.env.example` with API key set
2. ✅ **Prompt files created** (modifiable .txt files for each agent):
   - `prompts/resume_analyzer.txt` — expert resume extraction prompt
   - `prompts/jd_analyzer.txt` — expert JD extraction prompt
   - `prompts/strategy_agent.txt` — interview strategy + gap analysis prompt
3. ✅ **Prompt loader module** — `prompts/prompts.py` exposes RESUME_ANALYZER_PROMPT, JD_ANALYZER_PROMPT, STRATEGY_AGENT_PROMPT
4. ✅ **Three agents implemented**:
   - `agents/resume_analyzer.py` — extracts ResumeProfile with structured output
   - `agents/jd_analyzer.py` — extracts JDProfile with structured output
   - `agents/strategy_agent.py` — generates InterviewStrategy from resume + JD gap analysis
5. ✅ **Graph nodes** — `graph/nodes.py` with 3 node functions (analyze_resume_node, analyze_jd_node, build_strategy_node)
6. ✅ **LangGraph wiring** — `graph/interview_graph.py` with parallel fan-out/fan-in:
   - START → analyze_resume ∥ analyze_jd (parallel)
   - analyze_resume + analyze_jd → build_strategy (fan-in)
   - build_strategy → END
7. ✅ **Test suite**: `tests/test_phase2.py` — all 4 tests passing:
   - Resume analyzer produces ResumeProfile ✅
   - JD analyzer produces JDProfile ✅
   - Strategy agent generates InterviewStrategy ✅
   - Graph executes with parallel fan-out/fan-in ✅

**Test Results:**
```
============================= test session starts ==============================
collected 4 items

test_graph_parallel_fan_out_fan_in PASSED
test_jd_analyzer_returns_structured_profile PASSED
test_resume_analyzer_returns_structured_profile PASSED
test_strategy_agent_generates_interview_strategy PASSED

============================== 4 passed in 96.49s =============================
```

**Next:** Phase 3 implements the interview loop with Evaluator ∥ Question Planner → Conductor running in parallel per turn.

---

## Phase 3: Interview Loop — 3 Parallel Agents (Chat Mode)

**Goal:** The core loop with Evaluator ∥ Question Planner → Conductor. Chat mode only.

### Tasks

1. **`agents/evaluator.py`** — answer scoring agent:
   - `llm.with_structured_output(Evaluation)`
   - Can use a **cheaper/faster model** (`EVALUATOR_MODEL` from `.env`)
   - Input: question + user answer + optional code + difficulty + tier
   - Output: `Evaluation` with score, strengths, weaknesses, missed points, weak signals
   - For coding questions: includes `CodeFeedback` (correctness, complexity, edge cases)

2. **`agents/question_planner.py`** — adaptive next-question agent:
   - `llm.with_structured_output(QuestionPlan)`
   - Input: all `QuestionRecord`s so far, `InterviewStrategy`, type coverage, type scores, available types, tier, difficulty, topics already asked
   - Output: `QuestionPlan` — what type, what topic, is it a follow-up, directive for Conductor, resume hook
   - Logic encoded in the prompt:
     - First question: behavioral + resume-based (opening from `InterviewStrategy`)
     - Coverage enforcement: uncovered types get priority
     - Weak area weighting: avg score < 5 → lean into it
     - Follow-up rules: score < 4 → follow-up; 4-6 → 50/50; > 6 → new topic
     - Max follow-up depth: student=1, junior=2, senior=3
     - Difficulty ramping: 2 aces → harder; 2 fails → easier
     - Anti-repetition: check `topics_asked`
     - Resume hooks: use `InterviewStrategy.question_hooks` in priority order

3. **`agents/conductor.py`** — the interviewer voice:
   - `llm.with_structured_output(ConductorResponse)`
   - Input: `Evaluation` (from Evaluator) + `QuestionPlan` (from Planner) + conversation history + strategy + tier/feedback mode
   - Output: `ConductorResponse` — the natural-language message the user sees/hears
   - Responsibilities:
     - Deliver feedback if `feedback_mode = "after_each"` (uses Evaluation)
     - Transition naturally to next question (uses QuestionPlan)
     - Maintain tone per tier (encouraging / professional / challenging)
     - For coding questions: set `is_coding_question=True` and `question_text_for_display`

4. **`prompts/conductor.py`** — dynamic system prompt:
   ```python
   def build_conductor_prompt(tier, company, company_profile, feedback_mode) -> str:
   ```
   - Tier-specific tone and follow-up rules
   - Company-specific style adjustments
   - Feedback mode instructions

5. **`prompts/evaluator.py`** and **`prompts/question_planner.py`** — system prompts.

6. **Extend `graph/interview_graph.py`** — full loop with parallel fan-out:
   ```python
   # ── After setup ──
   builder.add_edge("build_strategy", "generate_first_question")
   builder.add_edge("generate_first_question", "wait_for_input")

   # ── Interview loop ──
   # Fan-out: user answer goes to Evaluator ∥ Planner simultaneously
   builder.add_edge("wait_for_input", "evaluate_answer")
   builder.add_edge("wait_for_input", "plan_next_question")

   # Fan-in: both must complete before Conductor speaks
   builder.add_edge("evaluate_answer", "conduct_turn")
   builder.add_edge("plan_next_question", "conduct_turn")

   builder.add_edge("conduct_turn", "check_continue")

   builder.add_conditional_edges(
       "check_continue",
       should_continue,
       {"continue": "wait_for_input", "end": "generate_report"}
   )
   ```
   - `interrupt_before=["wait_for_input"]` pauses graph for user input

7. **`graph/nodes.py`** — interview node functions:
   ```python
   def evaluate_answer_node(state: GrillMeState) -> dict:
       evaluation = evaluator.evaluate(
           question=state["question_records"][-1].question_text,
           answer=state["user_answer"],
           code=state["user_code"],
           difficulty=state["difficulty"],
           tier=state["experience_tier"]
       )
       return {"current_evaluation": evaluation}

   def plan_next_question_node(state: GrillMeState) -> dict:
       plan = question_planner.plan(
           records=state["question_records"],
           strategy=state["interview_strategy"],
           type_coverage=state["type_coverage"],
           type_scores=state["type_scores"],
           topics_asked=state["topics_asked"],
           available_types=state["question_types"],
           tier=state["experience_tier"],
           difficulty=state["difficulty"],
           follow_up_depth=state["follow_up_depth"]
       )
       return {"next_question_plan": plan}

   def conduct_turn_node(state: GrillMeState) -> dict:
       # Conductor sees BOTH evaluation and plan (fan-in)
       response = conductor.respond(
           evaluation=state["current_evaluation"],
           question_plan=state["next_question_plan"],
           conversation_history=state["conversation_history"],
           strategy=state["interview_strategy"],
           feedback_mode=state["feedback_mode"]
       )
       # Update state
       record = QuestionRecord(
           question_text=response.question_text_for_display or response.interviewer_message,
           question_type=state["next_question_plan"].question_type,
           question_topic=state["next_question_plan"].topic,
           user_answer=state["user_answer"],
           user_code=state["user_code"],
           evaluation=state["current_evaluation"],
           is_follow_up=state["next_question_plan"].is_follow_up,
           turn_number=state["question_count"] + 1
       )
       return {
           "agent_response": response.interviewer_message,
           "question_records": [*state["question_records"], record],
           "question_count": state["question_count"] + 1,
           "type_coverage": updated_coverage,
           "type_scores": updated_scores,
           "topics_asked": [*state["topics_asked"], state["next_question_plan"].topic],
           "conversation_history": updated_history,
       }
   ```

8. **`check_continue_node`**: checks elapsed time (45-min nudge), checks `should_end` flag.

9. **`ui/grind_page.py`** — chat UI:
   - `st.chat_message` for conversation display
   - Text input for answers
   - Sidebar: elapsed time, question count, type coverage stats, "End Grind" button
   - If feedback mode = "after_each": show evaluation (score + brief feedback) after each turn
   - 45-min nudge: dialog with continue/stop

10. **Conversation history management**:
    - Full history in state; summarize when > 20 messages
    - Only Conductor sees full history; Evaluator and Planner get condensed context

### Per-Turn Data Flow

```
User types answer (+ optional code)
         │
         ├──────────────────────┬──────────────────────────┐
         ▼                      ▼                          │
   ┌─────────────┐      ┌──────────────────┐               │
   │  Evaluator   │      │ Question Planner │               │
   │              │      │                  │               │
   │  Sees:       │      │  Sees:           │               │
   │  - question  │      │  - all records   │               │
   │  - answer    │      │  - strategy      │               │
   │  - code      │      │  - coverage      │               │
   │  - rubric    │      │  - scores        │               │
   │              │      │  - topics asked  │               │
   │  Returns:    │      │                  │               │
   │  Evaluation  │      │  Returns:        │               │
   │              │      │  QuestionPlan    │               │
   └──────┬──────┘      └───────┬──────────┘               │
          │                      │                          │
          └──────────┬───────────┘                          │
                     ▼                                      │
              ┌──────────────┐                              │
              │  Conductor    │                              │
              │               │                              │
              │  Sees:        │                              │
              │  - Evaluation │                              │
              │  - QuestionPlan                              │
              │  - conversation history                      │
              │  - strategy   │                              │
              │               │                              │
              │  Returns:     │                              │
              │  ConductorResponse (what user sees)          │
              └──────────────┘                              │
                     │                                      │
                     ▼                                      │
              [Display to user]                             │
              [Wait for next answer] ◀──────────────────────┘
```

### Exit Criteria
- Full chat loop: user answer → Evaluator ∥ Planner (parallel) → Conductor → display.
- Evaluator and Planner run simultaneously (observable via timing/logs).
- First question is resume-based behavioral.
- Questions mix types, lean into weak areas.
- 45-min nudge and "End Grind" work.
- All outputs are valid Pydantic objects.

---

## Phase 4: Voice Mode (STT + TTS)

**Goal:** User speaks answers, hears questions — toggle anytime.

### Tasks

1. **`voice/stt.py`** — faster-whisper:
   - `WhisperSTT` class, lazy model loading
   - `transcribe(audio_bytes) -> str`
   - Model size from `.env` (default `base`)

2. **`voice/tts.py`** — edge-tts:
   - `speak_sync(text, voice) -> bytes`
   - Voice from `.env` (default `en-US-AriaNeural`)

3. **Update `ui/grind_page.py`**:
   - `audio-recorder-streamlit` for mic recording
   - Flow: Record → Whisper → show transcript → user confirms/edits → submit to graph
   - Agent response → edge-tts → `st.audio()` autoplay
   - Mode toggle: voice ↔ chat, switchable mid-session
   - Coding questions in voice mode: speak + show text + code area

### Exit Criteria
- Speak → transcribe → agent responds with audio.
- Switch modes mid-session.
- Coding questions show text + code area in voice mode.

---

## Phase 5: Code Input for Coding Questions

**Goal:** Code text area appears for coding questions; Evaluator agent handles code evaluation.

### Tasks

1. **Update `ui/grind_page.py`**:
   - Detect `is_coding_question` from last Conductor response
   - Show `st.text_area` with monospace CSS
   - Question text displayed above code area
   - Submit sends `user_answer` + `user_code`

2. **Evaluator agent already handles code** (Phase 3): `CodeFeedback` in `Evaluation`.

3. **Conductor handles code follow-ups**: "Your solution is O(n^2). Can you optimize?" — code area stays visible.

4. **Code display CSS**: monospace, dark background.

### Exit Criteria
- Coding question → code area appears.
- Code evaluated (correctness, complexity).
- Follow-up coding → code area persists.
- Non-coding → no code area.

---


1. **Refine `prompts/question_planner.py`**:

2. **Refine `prompts/conductor.py`** — tier tone:
   - Student: "Good thinking! Let me push you a bit further..."
   - Junior: "That's partially correct. You're missing..."
   - Senior: "I'm not convinced. Your design has a single point of failure..."

3. **Refine `prompts/evaluator.py`** — tier-appropriate scoring calibration.

4. **Resume item tracking**: mark which `question_hooks` have been used → Planner avoids repeats.

### Exit Criteria
- 10+ question session shows natural mixing and adaptation.
- Weak areas get more questions.
- Follow-up depth matches tier.
- Resume-specific questions, not generic.
- No repeated topics.

---

## Phase 7: Report Generator + Resume Advisor

**Goal:** Comprehensive post-grind report. Also: resume improvement advice feature.

### Tasks

1. **`agents/report_generator.py`**:
   - `llm.with_structured_output(GrindReport)`
   - Two-pass:
     - Pass 1: per-question detail + category scores
     - Pass 2: weakness patterns + recommendations
   - Overall score: weighted average

2. **Wire into graph**: `generate_report` node → END.

3. **`ui/report_page.py`**:
   - Overall score (`st.metric`), category breakdown (colored cards)
   - Per-question `st.expander`: question, answer summary, score, strengths/weaknesses, code
   - Weakness patterns (`st.warning`), recommendations (numbered list)
   - Full transcript, markdown download

4. **Resume Advice feature** (reuses Strategy Agent):
   - `ui/resume_advice_page.py`: button "Get Resume Advice for this JD"
   - Calls `strategy_agent.generate_resume_advice(resume, jd)` → `ResumeAdvice`
   - Displays: skills to add, bullets to rewrite, gaps to address, fit score
   - Available from setup page (before grind) OR from report page (after grind)

### Exit Criteria
- Report generates with all sections populated.
- Weakness patterns are specific and evidence-backed.
- Resume advice shows actionable, JD-tailored suggestions.
- Markdown download works.

---

## Phase 8: Polish, Error Handling, Testing

**Goal:** Production-quality reliability and UX.

### Tasks

1. **Error handling**:
   - LLM failures: `with_retry()` (2 retries, exponential backoff)
   - Pydantic failures: `method="json_mode"` fallback
   - PDF parse errors: user-friendly message
   - Voice failures: fallback to chat
   - Context overflow: summarize at 20 messages

2. **UX polish**:
   - Spinners, progress indicators
   - Smooth page transitions
   - LangGraph `MemorySaver` checkpointing for session recovery
   - Dark mode compatible styling

3. **Prompt engineering iteration**:
   - Test all 3 tiers with real resumes
   - Verify tone differences across tiers
   - Test company-specific adjustments
   - Verify Gemini structured output reliability

4. **Testing**:
   - End-to-end grind (10+ questions) per tier
   - Voice roundtrip
   - Code evaluation quality
   - Report accuracy
   - Multi-model verification (Flash, Pro)

5. **Documentation**: `.env.example`, README.

### Exit Criteria
- Full session without crashes on all 3 tiers.
- Voice and chat both work.
- Report is meaningful.
- Errors show friendly messages.
- 2+ Gemini models verified.

---

## Phase Dependency Graph

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──────→ Phase 4 (voice)
 (scaffold    (parallel     (3 parallel    │
  + LangChain  setup         interview     ├──→ Phase 5 (code input)
  + Pydantic)  agents)       agents +      │
                             graph loop)   └──→ Phase 6 (adaptive refinement)
                                                    │
                                                    └──→ Phase 7 (report + resume advice)
                                                            │
                                                            └──→ Phase 8 (polish)
```

- Phases 4, 5, 6 can run in parallel after Phase 3.
- Phase 7 needs Phase 6 (finalized scoring).
- Phase 8 is final.

---

## Estimated Effort

| Phase | Effort | Cumulative |
|---|---|---|
| Phase 1: Scaffolding + LangChain + Pydantic models | ~3-4 hours | 3-4 hrs |
| Phase 2: Parallel setup agents (resume ∥ JD → strategy) | ~4-5 hours | 7-9 hrs |
| Phase 3: 3-agent interview loop (evaluator ∥ planner → conductor) | ~7-9 hours | 14-18 hrs |
| Phase 4: Voice mode (whisper + edge-tts) | ~4-5 hours | 18-23 hrs |
| Phase 5: Code input + evaluation | ~2-3 hours | 20-26 hrs |
| Phase 6: Adaptive refinement | ~3-4 hours | 23-30 hrs |
| Phase 7: Report + resume advisor | ~4-5 hours | 27-35 hrs |
| Phase 8: Polish + testing | ~3-4 hours | 30-39 hrs |
| **Total** | **~30-39 hours** | |

---

## Key Design Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Setup parallelism | Resume ∥ JD → Strategy (fan-out/fan-in) | No dependency between resume + JD parsing; ~33% faster |
| 2 | Interview parallelism | Evaluator ∥ Planner → Conductor | Evaluator and Planner are independent; Conductor needs both |
| 3 | Evaluator on cheaper model | Configurable via `EVALUATOR_MODEL` | Scoring is simpler than question generation — save cost/latency |
| 4 | Strategy Agent dual-purpose | Interview strategy + resume advice | Same gap analysis powers both features; justifies dedicated agent |
| 5 | Conductor as separate agent | Doesn't evaluate or plan — just speaks | Clean separation: scoring ≠ planning ≠ delivery |
| 6 | Question Planner as LLM agent (not pure code) | LLM-based with strategy context | Needs to reason about resume hooks, gap priorities, tone — not just weights |
| 7 | State machine | LangGraph with interrupt + checkpoint | Pause for user input, resume on submit, recoverable |
| 8 | Voice stack | faster-whisper + edge-tts | Both free, no API keys |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Gemini + `with_structured_output` unreliable | Pydantic validation fails | `method="json_mode"` fallback; `with_retry()`; test multiple models |
| Parallel fan-out latency mismatch | Faster agent waits for slower | Acceptable — still faster than sequential; Conductor only proceeds when both done |
| 3 LLM calls per turn = high cost | API bills | Evaluator on cheaper model; Planner prompts kept small; batch session limits |
| faster-whisper slow on CPU | Voice mode laggy | Default `base` model; show "Listening..." state |
| Conductor tone drift over long sessions | Stops matching tier | Re-inject tier instructions in every Conductor call (not just first) |
| Strategy Agent outputs low-quality hooks | Generic questions despite resume | Iterate on strategy prompt; include concrete examples in prompt |
