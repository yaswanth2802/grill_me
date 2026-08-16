# """Grind Page — Interview UI for GrillMe."""

# import streamlit as st
# from grillme.graph.interview_graph import interview_graph
# from grillme.models.state import GrillMeState


# def render_grind_page():
#     """Render the main interview grind page."""
#     st.header(f"🔥 Grind Interview: {st.session_state.get('company', 'Unknown')}")
    
#     # Initialize interview state if not already started
#     if "grind_started" not in st.session_state:
#         st.session_state.grind_started = False
#         st.session_state.thread_id = None
#         st.session_state.interview_state = None
    
#     if not st.session_state.grind_started:
#         # Start interview from setup data
#         if "resume_text" not in st.session_state or "jd_text" not in st.session_state:
#             st.error("Please complete the setup first!")
#             return
        
#         with st.spinner("⏳ Running setup: analyzing resume and JD..."):
#             # Initialize interview state from form data
#             initial_state: GrillMeState = {
#                 "resume_text": st.session_state.resume_text,
#                 "jd_text": st.session_state.jd_text,
#                 "company": st.session_state.get("company", "Unknown"),
#                 "experience_tier": st.session_state.get("experience_tier", "junior"),
#                 "difficulty": st.session_state.get("difficulty", "medium"),
#                 "question_types": st.session_state.get("question_types", ["behavioral", "technical", "system_design"]),
#                 "feedback_mode": st.session_state.get("feedback_mode", "after_each"),
#                 "interaction_mode": "chat",  # Phase 3 is chat-only
#                 # Initialize interview tracking
#                 "conversation_history": [],
#                 "question_records": [],
#                 "current_question_type": None,
#                 "type_coverage": {},
#                 "type_scores": {},
#                 "weak_areas": [],
#                 "topics_asked": [],
#                 "follow_up_depth": 0,
#                 "question_count": 0,
#                 "should_end": False,
#             }
            
#             # Create thread for checkpointing
#             thread_id = "grind-" + str(hash(st.session_state.resume_text))[:8]
#             config = {"configurable": {"thread_id": thread_id}}
            
#             # Run setup phase (analysis)
#             st.session_state.interview_state = interview_graph.invoke(initial_state, config=config)
#             st.session_state.grind_started = True
#             st.session_state.thread_id = thread_id
#             st.success("✅ Setup complete! Interview strategy loaded.")
#             st.rerun()
    
#     if st.session_state.grind_started:
#         # Display interview session
#         state = st.session_state.interview_state
        
#         # Conversation history
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.subheader(f"Question {state.get('question_count', 0)}")
#         with col2:
#             max_questions = 15
#             progress = state.get('question_count', 0) / max_questions
#             st.progress(progress, text=f"{state.get('question_count', 0)}/{max_questions}")
        
#         # Display conversation
#         conversation = state.get("conversation_history", [])
#         for msg in conversation:
#             role = msg.get("role", "unknown")
#             content = msg.get("content", "")
#             if role == "interviewer":
#                 with st.chat_message("assistant"):
#                     st.write(content)
#             elif role == "candidate":
#                 with st.chat_message("user"):
#                     st.write(content)
        
#         # Check if waiting for input
#         # (This is a simplified version; in production, you'd check LangGraph interrupt state)
#         show_input_form = True
        
#         if show_input_form:
#             st.divider()
#             st.subheader("Your Answer")
            
#             # Text answer
#             user_answer = st.text_area(
#                 "Type your answer here:",
#                 key="answer_input",
#                 height=100,
#                 placeholder="Provide a thoughtful, detailed answer..."
#             )
            
#             # Code submission (if applicable)
#             question_type = state.get("current_question_type", "behavioral")
#             user_code = None
#             if question_type in ["coding", "system_design"]:
#                 user_code = st.text_area(
#                     "Code (if applicable):",
#                     key="code_input",
#                     height=150,
#                     language="python",
#                     placeholder="def solution():\n    pass"
#                 )
            
#             # Submit answer button
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("📤 Submit Answer", key="submit_btn"):
#                     if not user_answer.strip():
#                         st.error("Please provide an answer before submitting.")
#                     else:
#                         # Add user answer to conversation
#                         conversation.append({
#                             "role": "candidate",
#                             "content": user_answer
#                         })
                        
#                         # Continue graph with user input
#                         with st.spinner("⏳ Evaluating answer..."):
#                             config = {"configurable": {"thread_id": st.session_state.thread_id}}
#                             input_state = {
#                                 **state,
#                                 "conversation_history": conversation,
#                                 "user_answer": user_answer,
#                                 "user_code": user_code,
#                                 "should_end": False,
#                                 "question_count": state.get("question_count", 0) + 1,
#                             }
                            
#                             # Resume graph (runs from wait_for_input interrupt)
#                             result_state = interview_graph.invoke(input_state, config=config)
                            
#                             # Add conductor response to conversation
#                             if result_state.get("agent_response"):
#                                 conductor_response = result_state["agent_response"].response_text
#                                 conversation.append({
#                                     "role": "interviewer",
#                                     "content": conductor_response
#                                 })
                            
#                             st.session_state.interview_state = result_state
#                             st.rerun()
            
#             with col2:
#                 if st.button("🛑 End Grind", key="end_btn"):
#                     st.session_state.interview_state["should_end"] = True
#                     st.info("Generating final report...")
#                     st.rerun()
        
#         # Display feedback summary (if available)
#         if state.get("current_evaluation"):
#             evaluation = state["current_evaluation"]
#             st.divider()
#             st.subheader("📊 Evaluation")
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Score", f"{evaluation.score}/10")
#             with col2:
#                 st.metric("Strengths", len(evaluation.strengths))
#             with col3:
#                 st.metric("Areas to Improve", len(evaluation.weaknesses))
            
#             if st.checkbox("Show detailed feedback", key="show_feedback"):
#                 st.write("**Strengths:**")
#                 for strength in evaluation.strengths:
#                     st.write(f"✅ {strength}")
                
#                 st.write("**Areas to Improve:**")
#                 for weakness in evaluation.weaknesses:
#                     st.write(f"⚠️ {weakness}")
                
#                 if evaluation.missed_points:
#                     st.write("**Missed Points:**")
#                     for point in evaluation.missed_points:
#                         st.write(f"💡 {point}")
"""Grind Page — Interview UI for GrillMe."""

import streamlit as st
from grillme.graph.interview_graph import interview_graph
from grillme.models.state import GrillMeState


def render_grind_page():
    """Render the main interview grind page."""
    st.header(f"🔥 Grind Interview: {st.session_state.get('company', 'Unknown')}")
    
    # Initialize interview state if not already started
    if "grind_started" not in st.session_state:
        st.session_state.grind_started = False
        st.session_state.thread_id = None
        st.session_state.interview_state = None
    
    if not st.session_state.grind_started:
        if "resume_text" not in st.session_state or "jd_text" not in st.session_state:
            st.error("Please complete the setup first!")
            return
        
        with st.spinner("⏳ Running setup: analyzing resume and JD..."):
            initial_state: GrillMeState = {
                "resume_text": st.session_state.resume_text,
                "jd_text": st.session_state.jd_text,
                "company": st.session_state.get("company", "Unknown"),
                "experience_tier": st.session_state.get("experience_tier", "junior"),
                "difficulty": st.session_state.get("difficulty", "medium"),
                "question_types": st.session_state.get("question_types", ["behavioral", "technical", "system_design"]),
                "feedback_mode": st.session_state.get("feedback_mode", "after_each"),
                "interaction_mode": "chat",
                "conversation_history": [],
                "question_records": [],
                "current_question_type": None,
                "type_coverage": {},
                "type_scores": {},
                "weak_areas": [],
                "topics_asked": [],
                "follow_up_depth": 0,
                "question_count": 0,
                "should_end": False,
            }
            
            thread_id = "grind-" + str(hash(st.session_state.resume_text))[:8]
            config = {"configurable": {"thread_id": thread_id}}
            
            # Initial run to start graph and hit first interrupt
            result_state = interview_graph.invoke(initial_state, config=config)
            
            if not result_state.get("conversation_history"):
                initial_response = result_state.get("agent_response")
                if initial_response:
                    raw_text = initial_response.response_text
                    first_q_text = "".join([item.get("text", "") for item in raw_text]) if isinstance(raw_text, list) else str(raw_text)
                else:
                    first_q_text = "Let's begin the interview. Tell me about your background."
                
                result_state["conversation_history"] = [
                    {"role": "interviewer", "content": first_q_text}
                ]
            
            st.session_state.interview_state = result_state
            st.session_state.grind_started = True
            st.session_state.thread_id = thread_id
            st.success("✅ Setup complete! Interview strategy loaded.")
            st.rerun()
            
    
    if st.session_state.grind_started:
        state = st.session_state.interview_state
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Question {state.get('question_count', 0) + 1}")
        with col2:
            max_questions = 15
            progress = min(state.get('question_count', 0) / max_questions, 1.0)
            st.progress(progress, text=f"{state.get('question_count', 0)}/{max_questions}")
        
        # Display conversation history safely from state
        conversation = state.get("conversation_history", [])
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "interviewer":
                with st.chat_message("assistant"):
                    st.write(content)
            elif role == "candidate":
                with st.chat_message("user"):
                    st.write(content)
        
        st.divider()
        
        # Use an st.form to cleanly batch inputs and avoid state desync on submit
        question_type = state.get("current_question_type", "behavioral")
        
        with st.form(key="answer_form", clear_on_submit=True):
            st.subheader("Your Answer")
            user_answer = st.text_area(
                "Type your answer here:",
                height=100,
                placeholder="Provide a thoughtful, detailed answer..."
            )
            
            user_code = None
            if question_type in ["coding", "system_design"]:
                user_code = st.text_area(
                    "Code (if applicable):",
                    height=150,
                    placeholder="def solution():\n    pass"
                )
            
            submitted = st.form_submit_button("📤 Submit Answer")
            
        if submitted:
            if not user_answer.strip():
                st.error("Please provide an answer before submitting.")
            else:
                with st.spinner("⏳ Evaluating answer & getting next question..."):
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    input_update = {
                        "user_answer": user_answer,
                        "user_code": user_code,
                        "should_end": False,
                        "question_count": state.get("question_count", 1) + 1,
                    }
                    
                    from langgraph.types import Command
                    result_state = interview_graph.invoke(Command(resume=input_update), config=config)
                    
                    # Extract the new interviewer message
                    new_agent_resp = result_state.get("agent_response")
                    next_q_text = "Let's continue."
                    if new_agent_resp:
                        if hasattr(new_agent_resp, "response_text"):
                            next_q_text = new_agent_resp.response_text
                        elif isinstance(new_agent_resp, dict):
                            next_q_text = new_agent_resp.get("response_text", next_q_text)

                    # Append to conversation history so it renders in chat bubbles
                    history = result_state.get("conversation_history", [])
                    history.append({"role": "candidate", "content": user_answer})
                    history.append({"role": "interviewer", "content": next_q_text})
                    result_state["conversation_history"] = history
                    
                    st.session_state.interview_state = result_state
                    st.rerun()

        if st.button("🛑 End Grind", key="end_btn"):
            if st.session_state.get("interview_state"):
                st.session_state.interview_state["should_end"] = True
            st.info("Generating final report...")
            st.rerun()
        
        # Display feedback summary (if available)
        if state.get("current_evaluation"):
            evaluation = state["current_evaluation"]
            st.divider()
            st.subheader("📊 Evaluation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{evaluation.score}/10")
            with col2:
                st.metric("Strengths", len(evaluation.strengths))
            with col3:
                st.metric("Areas to Improve", len(evaluation.weaknesses))
            
            if st.checkbox("Show detailed feedback", key="show_feedback"):
                st.write("**Strengths:**")
                for strength in evaluation.strengths:
                    st.write(f"✅ {strength}")
                
                st.write("**Areas to Improve:**")
                for weakness in evaluation.weaknesses:
                    st.write(f"⚠️ {weakness}")
                
                if evaluation.missed_points:
                    st.write("**Missed Points:**")
                    for point in evaluation.missed_points:
                        st.write(f"💡 {point}")