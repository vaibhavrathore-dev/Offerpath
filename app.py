import streamlit as st
from parser import extract_text_from_pdf
from analyzer import get_match_score, get_missing_skills, get_matched_skills
from llm import get_improvement_suggestions, get_resume_score_breakdown
from chatbot import get_chatbot_response
from guide import GUIDE_SECTIONS

st.set_page_config(page_title="OfferPath", page_icon="🎯", layout="wide")

# Initialize session state
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Analyzer", "Chatbot", "Resume Guide"])

# ── PAGE 1: ANALYZER ────────────────────────────────────────────────
if page == "Analyzer":
    st.title("🎯 OfferPath — Resume Analyzer")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    with col2:
        job_description = st.text_area("Paste the job description", height=200)

    if uploaded_file and job_description:
        if st.button("Analyze Resume", type="primary"):

            with st.spinner("Extracting text..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                st.session_state.resume_text = resume_text

            with st.spinner("Calculating score..."):
                score = get_match_score(resume_text, job_description)
                missing = get_missing_skills(resume_text, job_description)
                matched = get_matched_skills(resume_text, job_description)

            with st.spinner("Generating AI suggestions..."):
                suggestions = get_improvement_suggestions(resume_text, job_description, missing)
                breakdown = get_resume_score_breakdown(resume_text)

            st.divider()

            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
            st.markdown(f"<h1 style='color:{color}'>{score}% Match</h1>", unsafe_allow_html=True)
            st.progress(score / 100)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("✅ Matched Skills")
                for s in matched:
                    st.markdown(f"- {s}")

            with col4:
                st.subheader("❌ Missing Skills")
                for s in missing:
                    st.markdown(f"- {s}")

            st.subheader("💡 AI Improvement Suggestions")
            st.markdown(suggestions)

            st.subheader("📊 Section Breakdown")
            st.markdown(breakdown)

# ── PAGE 2: CHATBOT ─────────────────────────────────────────────────
elif page == "Chatbot":
    st.title("💬 OfferPath Chatbot")

    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload and analyze your resume first.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask something about your resume...")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(
                        user_input,
                        st.session_state.chat_history[:-1],
                        st.session_state.resume_text, ""
                    )
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# ── PAGE 3: RESUME GUIDE ─────────────────────────────────────────────
elif page == "Resume Guide":
    st.title("📖 Resume Writing Guide")

    for section_name, content in GUIDE_SECTIONS.items():
        with st.expander(f"📌 {section_name} Section"):
            st.markdown(f"**What it is:** {content['what']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Do:**")
                for tip in content["do"]:
                    st.markdown(f"- {tip}")
            with c2:
                st.markdown("**❌ Don't:**")
                for tip in content["dont"]:
                    st.markdown(f"- {tip}")
            st.code(content["example"], language=None)