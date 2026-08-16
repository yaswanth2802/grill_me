"""Main entry point for GrillMe Streamlit App."""

import streamlit as st
from grillme.grind_page import render_grind_page

st.set_page_config(
    page_title="GrillMe Interview App",
    page_icon="🔥",
    layout="wide"
)

def main():
    st.sidebar.title("🔥 GrillMe Navigation")
    pages = ["Setup", "Grind", "Report", "Resume Advice"]
    selection = st.sidebar.radio("Go to", pages)

    if selection == "Setup":
        st.header("🎯 Interview Setup")
        st.write("Configure your target company, role, and input your documents to begin.")
        
        # Capture setup form data into session state
        st.session_state.company = st.text_input("Target Company", value=st.session_state.get("company", "AI Corp"))
        st.session_state.experience_tier = st.selectbox("Experience Tier", ["junior", "mid", "senior"], index=2)
        st.session_state.difficulty = st.selectbox("Interview Difficulty", ["easy", "medium", "hard"], index=1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.resume_text = st.text_area(
                "Paste Resume Text",
                value=st.session_state.get("resume_text", ""),
                height=250,
                placeholder="Paste your professional experience and skills here..."
            )
        with col2:
            st.session_state.jd_text = st.text_area(
                "Paste Job Description (JD)",
                value=st.session_state.get("jd_text", ""),
                height=250,
                placeholder="Paste target job description requirements here..."
            )
            
        if st.button("🚀 Save Setup & Start Grind", type="primary"):
            if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
                st.error("Please provide both Resume and Job Description text.")
            else:
                st.success("Setup saved successfully! Switch to the **Grind** tab from the sidebar to start your interview.")

    elif selection == "Grind":
        # Check if setup is completed
        if not st.session_state.get("resume_text") or not st.session_state.get("jd_text"):
            st.warning("⚠️ Please complete the **Setup** tab first by entering your resume and job description.")
        else:
            render_grind_page()

    elif selection == "Report":
        st.header("📊 Interview Final Report")
        if "interview_state" in st.session_state and st.session_state.get("interview_state"):
            state = st.session_state.interview_state
            st.write(f"Total Questions Asked: {state.get('question_count', 0)}")
            st.write("Review your overall performance metrics and feedback history here.")
        else:
            st.info("No active or completed interview session found. Complete a grind session first!")

    else:
        st.header("💡 Resume Advice")
        st.write("Tailored resume improvement recommendations targeting multi-national corporation (MNC) standards will appear here.")

if __name__ == "__main__":
    main()